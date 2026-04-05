"""
Owlix FastAPI backend — v4.1.0
================================
- OwlixChain initialised inside @app.lifespan (after port is bound)
- Uvicorn binds port 10000 immediately at startup
- No module-level heavy imports that could crash before port binding
"""

import logging
import traceback
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import Optional, Any
import uvicorn
import requests as _requests

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("owlix")

# ── Google Speech API key (optional) ──────────────────────────────────────
GOOGLE_SPEECH_API_KEY = os.getenv("GOOGLE_SPEECH_API_KEY", "")

# ── Global chain instance (set during lifespan startup) ───────────────────
owlix = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs AFTER uvicorn binds the port.
    Heavy initialisation (Groq client, ChromaDB, SerpAPI) happens here.
    """
    global owlix
    logger.info("==> Lifespan startup: initialising OwlixChain...")
    try:
        from chain import OwlixChain
        owlix = OwlixChain()
        logger.info("==> OwlixChain ready.")
    except Exception as exc:
        logger.error("==> OwlixChain init failed: %s\n%s", exc, traceback.format_exc())
        # Don't raise — let the app stay up so Render sees an open port.
        # Queries will return a 503 until chain is ready.
    yield
    # Shutdown
    logger.info("==> Lifespan shutdown.")


app = FastAPI(title="Owlix RAG API", version="4.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    audio_base64: str
    encoding:     str = "WEBM_OPUS"
    sample_rate:  int = 48000
    lang:         str = "en-US"


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
    followups:      list
    sources:        list
    metrics:              Optional[MetricsModel]      = None
    credibility_report:   Optional[CredibilityReport] = None
    resolved_query:       Optional[str]  = None
    clarification_needed: Optional[bool] = False
    detected_lang:        Optional[str]  = "en-US"
    tts_available:        Optional[bool] = True


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
    return {
        "status": "Owlix is running",
        "version": "4.1.0",
        "backend": "Groq",
        "chain_ready": owlix is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chain_ready":  owlix is not None,
        "llm_backend":  "groq",
        "tts_backend":  "browser-native",
        "stt_backend":  "google" if GOOGLE_SPEECH_API_KEY else "browser-native",
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if owlix is None:
        raise HTTPException(
            status_code=503,
            detail="Service is still initialising. Please try again in a few seconds."
        )

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
            result["metrics"] = {
                "approximate": True,
                "confidence_only": result.get("confidence", "Low"),
            }

        if not result.get("credibility_report"):
            result["credibility_report"] = {"confidence": result.get("confidence", "Low")}

        result["tts_available"] = True

        if not result.get("detected_lang"):
            result["detected_lang"] = "en-US"

        logger.info("Query handled successfully: %r", req.query)
        return result

    except RuntimeError as exc:
        logger.error("=== QUERY FAILED (RuntimeError) ===\n%s", traceback.format_exc())
        if _is_groq_rate_limit(exc):
            raise HTTPException(status_code=503, detail=f"Groq rate limit exceeded: {exc}") from exc
        if _is_groq_unavailable(exc):
            raise HTTPException(status_code=503, detail=f"Groq service unavailable: {exc}") from exc
        raise HTTPException(status_code=500, detail=f"Groq inference error: {exc}") from exc

    except _requests.exceptions.Timeout as exc:
        raise HTTPException(status_code=503, detail="Request timed out. Please try again.") from exc

    except _requests.exceptions.ConnectionError as exc:
        raise HTTPException(status_code=503, detail="Cannot reach external services.") from exc

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        logger.error("=== QUERY FAILED ===\n%s", traceback.format_exc())
        return _fallback_response(
            f"An unexpected error occurred: {type(exc).__name__}: {str(exc)}"
        )


# ── Speech-to-Text ─────────────────────────────────────────────────────────

@app.post("/speech-to-text")
async def speech_to_text(req: SpeechRequest):
    if not GOOGLE_SPEECH_API_KEY:
        return JSONResponse(status_code=200, content={
            "transcript": "",
            "error": "Google Speech API key not configured.",
            "fallback": True,
            "use_browser_stt": True,
        })

    try:
        url = f"https://speech.googleapis.com/v1/speech:recognize?key={GOOGLE_SPEECH_API_KEY}"
        encoding_map = {
            "WEBM_OPUS": "WEBM_OPUS", "OGG_OPUS": "OGG_OPUS",
            "LINEAR16": "LINEAR16",   "MP4": "MP4",
        }
        payload = {
            "config": {
                "encoding":        encoding_map.get(req.encoding.upper(), "WEBM_OPUS"),
                "sampleRateHertz": req.sample_rate,
                "languageCode":    req.lang or "en-US",
                "enableAutomaticPunctuation": True,
                "model": "latest_short",
            },
            "audio": {"content": req.audio_base64},
        }
        resp = _requests.post(url, json=payload, timeout=30)
        if resp.status_code != 200:
            return JSONResponse(status_code=200, content={
                "transcript": "", "fallback": True, "use_browser_stt": True,
                "error": f"Google Speech API error {resp.status_code}",
            })

        data    = resp.json()
        results = data.get("results", [])
        if not results:
            return JSONResponse(status_code=200, content={
                "transcript": "", "error": "No speech detected",
                "fallback": False, "use_browser_stt": False,
            })

        transcript = results[0].get("alternatives", [{}])[0].get("transcript", "")
        confidence = results[0].get("alternatives", [{}])[0].get("confidence", 0.0)
        return {
            "transcript": transcript.strip(), "confidence": confidence,
            "fallback": False, "use_browser_stt": False,
        }

    except Exception as exc:
        logger.error("Speech-to-text failed: %s", exc)
        return JSONResponse(status_code=200, content={
            "transcript": "", "error": str(exc),
            "fallback": True, "use_browser_stt": True,
        })


# ── TTS ────────────────────────────────────────────────────────────────────

@app.post("/tts")
async def tts_endpoint(request: Request):
    try:
        body = await request.json()
        text = (body.get("text") or "").strip()
        lang = (body.get("lang") or "en-US").strip()

        if not text:
            return JSONResponse(status_code=400, content={
                "error": "No text provided for TTS.", "tts_available": False,
            })

        text = clean_text_for_tts(text)
        MAX_TTS_CHARS = 4000
        if len(text) > MAX_TTS_CHARS:
            text = text[:MAX_TTS_CHARS] + ". End of response."

        if not text:
            return JSONResponse(status_code=400, content={
                "error": "Text is empty after cleaning.", "tts_available": False,
            })

        return {"text": text, "lang": lang or "en-US",
                "tts_available": True, "char_count": len(text)}

    except Exception as exc:
        logger.warning("TTS endpoint failed: %s", exc)
        return JSONResponse(status_code=200, content={
            "error": f"TTS failed: {exc}", "tts_available": False,
        })


@app.get("/tts/check")
async def tts_check():
    return {"tts_available": True, "mode": "browser-native-synthesis"}


# ── Memory ─────────────────────────────────────────────────────────────────

@app.delete("/memory/{session_id}")
def clear_memory(session_id: str):
    if owlix is None:
        raise HTTPException(status_code=503, detail="Service still initialising.")
    owlix.clear_memory(session_id)
    return {"status": "memory cleared", "session_id": session_id}


# ── Global exception handler ────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "detail": "AI processing failed. Please try again.",
            "type": type(exc).__name__,
        },
    )


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)