# Owlix AI — RAG Assistant (Full Fixed Build)

## Project Structure
```
owlix_project/
├── main.py              # FastAPI backend (Steps 10–12, global validation)
├── chain.py             # RAG pipeline (Steps 0–13, all fixed)
├── owlix_frontend.html  # Frontend UI — open directly in browser
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── chroma_db/           # Auto-created on first run (vector store)
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — add GEMINI_API_KEY and SERPAPI_API_KEY
```

### 3. Run backend
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Open frontend
Open `owlix_frontend.html` in your browser (Chrome recommended for full voice support).

---

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root health check |
| GET | `/health` | Status |
| POST | `/query` | Main RAG query endpoint |
| POST | `/tts` | Text-to-speech validation + truncation |
| DELETE | `/memory/{session_id}` | Clear session memory |

---

## 🐛 Critical Bug Fix — Response Disappears After ~2 Seconds

**Root cause:** `startNewSession()` created a session ID and stored it in `localStorage` as the "active" key, but **never inserted the session object into `state.sessions`**. As a result:

- `addMessage()` called `getSession(id)` which returned `null` → **silently dropped every message**
- The bot response was only in the DOM (rendered HTML), not in `state.sessions`
- Any re-render, scroll event (`scrollToBottom` timeout), or sidebar interaction wiped the chat area
- This made responses appear for ~2 seconds then vanish completely

**Fixes applied to `owlix_frontend.html`:**

1. **`startNewSession()`** — now calls `state.sessions.unshift(...)` + `saveSessions()` so the session is immediately persisted before any message is added.
2. **`sendMessage()`** — guard changed to `!getSession(state.currentSessionId)` (not just `!state.currentSessionId`) so a session that was cleared mid-flow is re-created correctly.
3. **`addMessage()`** — now auto-creates the session on-the-fly via `upsertSession` if `getSession` returns null, so messages are never silently dropped even in edge cases.
4. **`renderHistoryList()`** — filters to only sessions with at least one message, preventing blank ghost entries accumulating in the sidebar.
5. **`boot()`** — cleans up stale empty sessions from `localStorage` on startup, then creates a fresh session properly.
6. **`renderHistoryList()` called after bot response** — sidebar now updates correctly after each conversation turn so history is immediately visible.

---

## What Was Fixed (vs Previous Build)

### Step 0 — Empty Input UI Validation
- Frontend shows a toast: *"Please enter text or use voice input."* when Send is clicked with an empty box.

### Step 1 — Voice Input Error Messages
- Browser doesn't support Speech Recognition: *"Voice input is not supported in this browser. Please type your query."*
- No speech detected: *"No speech detected. Please try again or type your query."*
- Microphone denied: *"Microphone access denied. Please allow microphone in browser settings."*
- Other errors: *"Voice input could not be processed. Please try again or type your query."*
- On successful capture: info toast confirms the transcribed text.

### Step 1 — Preprocessing Auto-Clean Fallback
- `chain.py` `preprocess_query()` has 3 stages:
  1. Immediate rejection of fully blank input
  2. Auto-clean attempt (whitespace, punctuation, filler stripping) — wrapped in try/except so a crash falls back to the stripped raw input rather than crashing the pipeline
  3. Final check: only raises `ValueError` if nothing usable remains after recovery

### Step 11 — All Response Fields Displayed
- **Key Events** — important events or actions from sources
- **Contradictions** — conflicting information found across sources
- **Current Status** — latest situation based on retrieved data
- **Uncertainty** — missing information or ambiguity (including hallucination/bias warnings)

### Step 11 — Three Separate Score Panels
- **Confidence** — High/Medium/Low from the credibility framework (composite score %)
- **Relevance** — average of Precision + Recall, expressed as a percentage
- **Coverage** — number of sources actually used in the response

### Step 12 — TTS Voice Output (Speak Button)
Every bot response has a **Speak** button that:
1. Calls `/tts` backend to validate and truncate the text (max 4000 chars)
2. Uses browser `SpeechSynthesis` to read the response aloud
3. Button becomes **Stop** while speaking
4. If `/tts` returns `tts_available: false` → shows toast: *"Voice output not available. Showing text only."*
5. If browser `SpeechSynthesis` is unavailable → shows toast: *"Voice output is not supported in this browser."*
6. If speech fails mid-playback → shows toast: *"Voice output failed. Showing text only."*

---

## Environment Variables
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | — | Google Gemini API key |
| `SERPAPI_API_KEY` | Yes | — | SerpAPI key for web search |
| `CHROMA_PERSIST_DIR` | No | `./chroma_db` | ChromaDB storage path |
| `GEMINI_CHAT_MODEL` | No | `gemini-2.0-flash` | Gemini chat model |
| `GEMINI_EMBEDDING_MODEL` | No | `gemini-embedding-001` | Gemini embedding model |

---

## What Was Added — Doc Gap Fixes

### Step 1 — Multilingual STT
- `recognition.lang` is set from `detectedLang` (BCP-47 code returned by backend after each query), e.g. `hi-IN`, `fr-FR`, `ar-SA`.

### Step 1 — Energy-Based Noise Gating
- `getMicEnergy()` computes RMS amplitude. Transcripts with energy < 4 are discarded: *"Low audio signal detected. Please speak closer to your microphone."*

### Step 1 — Noise Filtering via Ranking
- `rank_query_noise(query)` in `chain.py` scores information density. Queries below 0.25 (pure filler) are rejected before reaching the LLM.

### Step 1 / Step 12 — Multilingual TTS
- `utter.lang` and the `/tts` `lang` field use `detectedLang`, so TTS speaks in the correct language.

### Step 7 — `detected_lang` in LLM Response
- System prompt instructs LLM to return `detected_lang` BCP-47 code in every JSON response.
- `QueryResponse` Pydantic model includes `detected_lang: Optional[str] = "en-US"`.
- Propagated end-to-end: LLM → backend → frontend → STT/TTS.