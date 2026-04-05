"""
Owlix FastAPI backend — Steps 10 / 11 / 12 / Global Validation
===============================================================
BACKEND: Groq Inference API (replaces Hugging Face)

CHANGES (v4.0.0):
  - Replaced HF_API_TOKEN with GROQ_API_KEY
  - Removed all HF-specific references from health endpoint
  - /speech-to-text returns proper fallback flag so frontend knows to use browser STT
  - CORS allows all origins for local dev
  - /tts endpoint validates and pre-processes text for browser SpeechSynthesis
"""


import logging
import traceback
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import Optional, Any
import uvicorn
import requests as _requests

from chain import OwlixChain

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("owlix")

app = FastAPI(title="Owlix RAG API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

owlix = OwlixChain()

# Google Speech API key from env (optional)
GOOGLE_SPEECH_API_KEY = os.getenv("GOOGLE_SPEECH_API_KEY", "")


# ── Request / Response models ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query cannot be empty. Please enter a valid question.")
        return v


class SpeechRequest(BaseModel):
    """Step 12: Voice input — base64-encoded audio + language code."""
    audio_base64: str
    encoding:     str  = "WEBM_OPUS"
    sample_rate:  int  = 48000
    lang:         str  = "en-US"


class TTSRequest(BaseModel):
    """Step 12: Text-to-Speech request."""
    text: str
    lang: Optional[str] = "en-US"


class MetricsModel(BaseModel):
    precision:       Optional[float] = None
    recall:          Optional[float] = None
    faithfulness:    Optional[float] = None
    coverage:        Optional[int]   = None
    approximate:     Optional[bool]  = True
    confidence_only: Optional[str]   = None


class CredibilityReport(BaseModel):
    confidence:         Optional[str]   = "Low"
    composite_score:    Optional[float] = None
    source_credibility: Optional[float] = None
    source_agreement:   Optional[float] = None
    consistency:        Optional[float] = None
    time_relevance:     Optional[float] = None
    bias_flag:          Optional[bool]  = False


class QueryResponse(BaseModel):
    summary:        str
    key_findings:   str
    key_events:     str
    contradictions: str
    current_status: str
    conclusion:     str
    uncertainty:    str
    confidence:     str
    followups:      list[str]
    sources:        list[dict]
    metrics:            Optional[MetricsModel]      = None
    credibility_report: Optional[CredibilityReport] = None
    resolved_query:       Optional[str]  = None
    clarification_needed: Optional[bool] = False
    detected_lang:        Optional[str]  = "en-US"
    tts_available: Optional[bool] = True


# ── Helpers ────────────────────────────────────────────────────────────────

def coerce_str(val: Any) -> str:
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    if val is None:
        return "—"
    return str(val)


def _fallback_response(message: str) -> dict:
    return {
        "summary":        message,
        "key_findings":   "—", "key_events": "—", "contradictions": "—",
        "current_status": "—", "conclusion": "—",
        "uncertainty":    message,
        "confidence":     "Low",
        "followups": ["Please try again.", "Try a different question.", "Check the backend logs."],
        "sources":              [],
        "resolved_query":       "",
        "clarification_needed": False,
        "metrics":              {"approximate": True},
        "credibility_report":   {"confidence": "Low"},
        "tts_available":        True,
    }


def _is_groq_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("429", "rate limit", "quota", "too many requests", "rate_limit"))


def _is_groq_unavailable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("503", "service unavailable"))


def clean_text_for_tts(text: str) -> str:
    """Strip markdown, URLs, and special symbols so TTS sounds natural."""
    import re
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[⚠★●◆▶►◀◁→←↑↓]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Owlix is running", "version": "4.0.0", "backend": "Groq"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_backend":  "groq",
        "tts_backend":  "browser-native",
        "stt_backend":  "google" if GOOGLE_SPEECH_API_KEY else "browser-native",
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    try:
        logger.info("Received query: %r | session: %s", req.query, req.session_id)
        result = await owlix.run(req.query, req.session_id)

        for field in (
            "summary", "key_findings", "key_events", "contradictions",
            "current_status", "conclusion", "uncertainty", "confidence",
        ):
            result[field] = coerce_str(result.get(field, "—"))

        followups = result.get("followups", [])
        if not isinstance(followups, list):
            followups = [str(followups)]
        result["followups"] = [str(f) for f in followups][:3]
        while len(result["followups"]) < 3:
            result["followups"].append("Tell me more.")

        if not isinstance(result.get("sources"), list):
            result["sources"] = []

        if not result.get("metrics"):
            result["metrics"] = {"approximate": True, "confidence_only": result.get("confidence", "Low")}

        if not result.get("credibility_report"):
            result["credibility_report"] = {"confidence": result.get("confidence", "Low")}

        result["tts_available"] = True

        if not result.get("detected_lang"):
            result["detected_lang"] = "en-US"

        logger.info("Query handled successfully: %r", req.query)
        return result

    except RuntimeError as exc:
        logger.error("=== QUERY FAILED (Groq RuntimeError) ===\n%s", traceback.format_exc())
        if _is_groq_rate_limit(exc):
            raise HTTPException(status_code=503, detail=f"Groq rate limit exceeded: {exc}") from exc
        if _is_groq_unavailable(exc):
            raise HTTPException(status_code=503, detail=f"Groq service unavailable, retry shortly: {exc}") from exc
        raise HTTPException(status_code=500, detail=f"Groq inference error: {exc}") from exc

    except _requests.exceptions.Timeout as exc:
        raise HTTPException(status_code=503, detail="Request timed out. Please try again.") from exc

    except _requests.exceptions.ConnectionError as exc:
        raise HTTPException(status_code=503, detail="Cannot reach external services.") from exc

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        logger.error("=== QUERY FAILED ===\n%s", traceback.format_exc())
        return _fallback_response(f"An unexpected error occurred: {type(exc).__name__}: {str(exc)}")


# ── Step 12: Google Speech-to-Text endpoint ────────────────────────────────
@app.post("/speech-to-text")
async def speech_to_text(req: SpeechRequest):
    """
    Convert base64-encoded audio to text using Google Cloud Speech-to-Text REST API.
    Falls back to browser Web Speech API if GOOGLE_SPEECH_API_KEY is not set.
    """
    if not GOOGLE_SPEECH_API_KEY:
        return JSONResponse(
            status_code=200,
            content={
                "transcript": "",
                "error": "Google Speech API key not configured. Set GOOGLE_SPEECH_API_KEY in .env",
                "fallback": True,
                "use_browser_stt": True,
            }
        )

    try:
        url = f"https://speech.googleapis.com/v1/speech:recognize?key={GOOGLE_SPEECH_API_KEY}"

        encoding_map = {
            "WEBM_OPUS": "WEBM_OPUS",
            "OGG_OPUS":  "OGG_OPUS",
            "LINEAR16":  "LINEAR16",
            "MP4":       "MP4",
        }
        google_encoding = encoding_map.get(req.encoding.upper(), "WEBM_OPUS")
        lang = req.lang if req.lang else "en-US"

        payload = {
            "config": {
                "encoding":        google_encoding,
                "sampleRateHertz": req.sample_rate,
                "languageCode":    lang,
                "enableAutomaticPunctuation": True,
                "model": "latest_short",
            },
            "audio": {
                "content": req.audio_base64
            }
        }

        resp = _requests.post(url, json=payload, timeout=30)

        if resp.status_code != 200:
            logger.error("Google Speech API error %d: %s", resp.status_code, resp.text[:200])
            return JSONResponse(
                status_code=200,
                content={
                    "transcript": "",
                    "error": f"Google Speech API error {resp.status_code}: {resp.text[:100]}",
                    "fallback": True,
                    "use_browser_stt": True,
                }
            )

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return JSONResponse(
                status_code=200,
                content={
                    "transcript": "",
                    "error": "No speech detected",
                    "fallback": False,
                    "use_browser_stt": False,
                }
            )

        transcript = results[0].get("alternatives", [{}])[0].get("transcript", "")
        confidence = results[0].get("alternatives", [{}])[0].get("confidence", 0.0)

        logger.info("Speech-to-text: %r (confidence: %.2f)", transcript, confidence)
        return {
            "transcript":     transcript.strip(),
            "confidence":     confidence,
            "fallback":       False,
            "use_browser_stt": False,
        }

    except Exception as exc:
        logger.error("Speech-to-text failed: %s", exc)
        return JSONResponse(
            status_code=200,
            content={
                "transcript":     "",
                "error":          str(exc),
                "fallback":       True,
                "use_browser_stt": True,
            }
        )


# ── Step 12: TTS endpoint ─────────────────────────────────────────────────
@app.post("/tts")
async def tts_endpoint(request: Request):
    """
    Text-to-Speech pre-processing endpoint.
    Cleans text and returns it for browser SpeechSynthesis.
    No audio binary is returned — synthesis happens client-side.
    """
    try:
        body = await request.json()
        text = (body.get("text") or "").strip()
        lang = (body.get("lang") or "en-US").strip()

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "No text provided for TTS.", "tts_available": False},
            )

        text = clean_text_for_tts(text)

        MAX_TTS_CHARS = 4000
        if len(text) > MAX_TTS_CHARS:
            text = text[:MAX_TTS_CHARS] + ". End of response."

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Text is empty after cleaning.", "tts_available": False},
            )

        return {
            "text":          text,
            "lang":          lang or "en-US",
            "tts_available": True,
            "char_count":    len(text),
        }

    except Exception as exc:
        logger.warning("TTS endpoint failed: %s", exc)
        return JSONResponse(
            status_code=200,
            content={"error": f"TTS failed: {exc}", "tts_available": False},
        )


@app.get("/tts/check")
async def tts_check():
    """Quick endpoint for frontend to verify TTS backend is available."""
    return {"tts_available": True, "mode": "browser-native-synthesis"}


# ── Memory management ───────────────────────────────────────────────────────
@app.delete("/memory/{session_id}")
def clear_memory(session_id: str):
    owlix.clear_memory(session_id)
    return {"status": "memory cleared", "session_id": session_id}


# ── Global exception handler ────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "AI processing failed. Please try again.", "type": type(exc).__name__},
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)