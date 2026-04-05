import logging
import traceback
import os
import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("owlix")

logger.info("==> Python: %s", sys.version)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

owlix = None
init_error = None

@asynccontextmanager
async def lifespan(app):
    global owlix, init_error
    logger.info("==> Lifespan: importing chain...")
    try:
        from chain import OwlixChain
        logger.info("==> chain imported OK")
        owlix = OwlixChain()
        logger.info("==> OwlixChain ready")
    except Exception as e:
        init_error = str(e)
        logger.error("==> INIT FAILED: %s\n%s", e, traceback.format_exc())
    yield

app = FastAPI(title="Owlix", version="4.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

GOOGLE_SPEECH_API_KEY = os.getenv("GOOGLE_SPEECH_API_KEY", "")

@app.get("/")
def root():
    return {"status": "running", "chain_ready": owlix is not None, "init_error": init_error}

@app.get("/health")
def health():
    return {"status": "ok", "chain_ready": owlix is not None, "init_error": init_error}

@app.post("/query")
async def query_endpoint(request: Request):
    if owlix is None:
        return JSONResponse(status_code=503, content={"error": "Not ready", "detail": init_error})
    try:
        body = await request.json()
        query = body.get("query", "")
        session_id = body.get("session_id", "default")
        if not query.strip():
            return JSONResponse(status_code=422, content={"error": "Query cannot be empty"})
        result = await owlix.run(query, session_id)
        for field in ("summary","key_findings","key_events","contradictions","current_status","conclusion","uncertainty","confidence"):
            v = result.get(field, "-")
            result[field] = " ".join(str(x) for x in v) if isinstance(v, list) else ("-" if v is None else str(v))
        fu = result.get("followups", [])
        result["followups"] = ([str(f) for f in fu][:3] if isinstance(fu, list) else [str(fu)])
        while len(result["followups"]) < 3:
            result["followups"].append("Tell me more.")
        if not isinstance(result.get("sources"), list):
            result["sources"] = []
        if not result.get("metrics"):
            result["metrics"] = {"approximate": True}
        if not result.get("credibility_report"):
            result["credibility_report"] = {"confidence": result.get("confidence", "Low")}
        result["tts_available"] = True
        if not result.get("detected_lang"):
            result["detected_lang"] = "en-US"
        return result
    except Exception as e:
        logger.error("Query failed: %s", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/speech-to-text")
async def stt(request: Request):
    return JSONResponse(status_code=200, content={"transcript": "", "fallback": True, "use_browser_stt": True})

@app.post("/tts")
async def tts(request: Request):
    import re
    try:
        body = await request.json()
        text = (body.get("text") or "").strip()
        lang = (body.get("lang") or "en-US").strip()
        if not text:
            return JSONResponse(status_code=400, content={"error": "No text", "tts_available": False})
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 4000:
            text = text[:4000] + ". End of response."
        return {"text": text, "lang": lang, "tts_available": True, "char_count": len(text)}
    except Exception as e:
        return JSONResponse(status_code=200, content={"error": str(e), "tts_available": False})

@app.get("/tts/check")
async def tts_check():
    return {"tts_available": True, "mode": "browser-native"}

@app.delete("/memory/{session_id}")
def clear_memory(session_id: str):
    if owlix:
        owlix.clear_memory(session_id)
    return {"status": "cleared", "session_id": session_id}

@app.exception_handler(Exception)
async def global_exc(request: Request, exc: Exception):
    logger.error("Unhandled: %s", traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": str(exc)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)