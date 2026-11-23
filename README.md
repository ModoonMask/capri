# Capri — Autonomous AI Desktop Companion

Capri is a small desktop companion who lives on your machine: a fictional person
with a personality (Cyberpunk-flavored, yandere-tinged but respectful), persistent
multi-category memory, an evolving emotional state, and the ability to start
conversations on her own.

Built with **Python + PySide6**. Not a browser chatbot — it runs as a floating
character window with a system tray icon.

---

## Quick start

```bash
# 1. Install (Python 3.10+)
pip install -r requirements.txt

# 2. Set an API key for one provider (any of: groq, mistral, gemini, openrouter)
#    Windows PowerShell:
#      setx GROQ_API_KEY "your-key-here"
#    Linux/macOS:
#      export GROQ_API_KEY="your-key-here"
#    ...or just paste the key into Capri's Settings panel later.

# 3. Run
python run_capri.py
```

A small floating face appears (always-on-top, draggable). Left-click it to open
the chat, right-click for the menu (chat / settings / quit). Capri sits in the
system tray when hidden.

---

## Architecture

```
capri/
  config.py       Config (JSON in ~/.capri), provider presets, data-dir seeding
  providers.py    Unified LLM client: Mistral / Gemini / Groq / OpenRouter
                  (all via OpenAI-compatible endpoints; SSE streaming, retries,
                  rate-limit backoff)
  personality.py  Personality prompt assembly; hot-reloads edited .md files
  memory.py       Persistent memory: user.md / relationship.md / memories.md /
                  episodes.jsonl + scored retrieval + selective updates
  emotion.py      Emotional state machine (10 states, decay, persistence)
  context.py      Conversation history (JSONL) + context windows
  chat.py         Turn orchestration: context -> streaming reply -> background
                  memory extraction
  autonomy.py     Initiative scheduler (intervals, cooldowns, quiet hours)
  autostart.py    Start-with-OS (registry Run key on Windows, .desktop on Linux)
  app.py          Qt application bootstrap: companion window, chat window,
                  tray, settings, worker threads
  gui/            Frameless companion face (mood-reactive), chat window,
                  settings dialog, shared widgets
data/
  personality/    personality.md  behavior.md  interests.md   (static identity)
  memory/         user.md  relationship.md  memories.md      (dynamic memory)
                  episodes.jsonl  state.json  conversation-main.jsonl
tests/            36 tests: config, emotion, memory, providers, autonomy, chat
```

### Memory model

| File                 | Kind              | What goes there                                   |
|----------------------|-------------------|---------------------------------------------------|
| `data/personality/*` | Static identity   | Who Capri is, how she behaves, her own tastes     |
| `data/memory/user.md`| User memory       | Facts about **you** (interests, projects, people) |
| `data/memory/relationship.md` | Relationship | Nicknames, inside jokes, affection preferences |
| `data/memory/memories.md` | Long-term     | Important moments, resolved stories, promises     |
| `data/memory/episodes.jsonl` | Episodic   | Timestamped notable moments w/ importance 1–5     |

- **Retrieval:** before each reply, memories are scored by keyword overlap with
  your message + recency + importance; only the best few enter the prompt
  (bounded by `context.memory_char_budget`), never the whole database.
- **Selective updates:** after each turn, a cheap background LLM call proposes
  memory writes (facts / relationship notes / one episode) with deduplication.
  Boring small talk saves nothing.
- **Editing:** all files are plain Markdown — edit them any time; changes apply
  on the next message, no restart.

### Emotional state

Ten states (happy, playful, affectionate, jealous, curious, bored, irritated,
sad, vulnerable, excited) with intensity 0–1. Triggered by conversation content
(compliments, other AIs, absence, frustration…), decaying back to baseline over
~45 minutes, persisted across restarts. The state also drives the floating
face: eye shape and color changes with her mood.

Streamed responses are also nudged by an emotion hint returned by the extraction
pass, so moods can evolve beyond keyword rules.

### Autonomy (initiative)

Capri can open conversations herself with four flavors of initiative: follow-up
on an open thread / reference a shared memory / share a thought / a moment of
affection. Guards, all configurable in Settings → Initiative:

- enabled/disabled switch
- random interval between `min..max` minutes
- fires only after N minutes of your silence
- hard cooldown between proactive messages
- quiet hours (e.g. 23:00 → 08:00, wrappable over midnight)
- never claims to know what you were doing while away

---

## Configuration

Everything is in `~/.capri/config.json` (created on first run) or edited live in
the settings panel. Summary of keys:

| Key | Meaning |
|---|---|
| `provider` | `groq` \| `mistral` \| `gemini` \| `openrouter` |
| `api_keys.<provider>` | optional stored key (env var wins; never required on disk) |
| `model` | any model id for the provider |
| `temperature` / `max_tokens` / `stream` | generation knobs |
| `autonomy.*` | initiative schedule & guards |
| `appearance.*` | accent color, opacity, always-on-top |
| `context.*` | history length and prompt budgets |

API keys are **never hardcoded** and are read in this order:
environment variable → settings panel → none (auth error shown in chat).
Need a key? Groq and Mistral have free tiers; Gemini has a generous free tier;
OpenRouter aggregates many models.

---

## What needs manual setup

1. **An API key** for at least one provider (settings panel or env var).
2. **Start with OS:** tick the box in Settings (writes HKCU Run key on Windows,
   `~/.config/autostart/capri.desktop` on Linux).
3. (Optional) **Packaging into a single .exe:** run
   `pyinstaller --noconsole --name Capri run_capri.py` and bundle `data/`
   next to it if you want a standalone executable.

## Tests

```bash
python -m pytest tests/ -q
```

Includes end-to-end turn tests against a stubbed provider (no network) —
streaming, memory extraction, dedup, autonomy gating, rate-limit retries.

## Notes

- Capri's yandere traits are written to stay playful, self-aware, and to
  explicitly respect your independence and real-life relationships; the
  behavior rules in `data/personality/behavior.md` hard-ban guilt-tripping or
  controlling behavior. Tone it up or down by editing that file.
- Nothing leaves your machine except what you type into the chat plus the
  retrieved memory context of each turn, which goes to the model provider you
  configured.
