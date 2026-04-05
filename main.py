"""
Owlix FastAPI backend — DIAGNOSTIC VERSION
This version starts instantly and logs all errors clearly.
"""

import logging
import traceback
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("owlix")

logger.info("==> Python version: %s", sys.version)
logger.info("==> Starting Owlix imports...")

from dotenv import load_dotenv
load_dotenv()
logger.info("==> dotenv loaded")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
logger.info("==> FastAPI imported")

owlix = None
init_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global owlix, init_error
    logger.info("==> Lifespan: beginning chain init...")

    try:
        logger.info("==> Step 1: importing chain module...")
        from chain import OwlixChain
        logger.info("==> Step 1: chain module imported OK")
    except Exception as e:
        init_error = f"Chain import failed: {e}\n{traceback.format_exc()}"
        logger.error("==> CHAIN IMPORT FAILED: %s", init_error)
        yield
        return

    try:
        logger.info("==> Step 2: creating OwlixChain instance...")
        owlix = OwlixChain()
        logger.info("==> Step 2: OwlixChain created OK")
    except Exception as e:
        init_error = f"OwlixChain init failed: {e}\n{traceback.format_exc()}"
        logger.error("==> OWLIX INIT FAILED: %s", init_error)
        yield
        return

    logger.info("==> All init complete, app is ready!")
    yield
    logger.info("==> Shutdown")


app = FastAPI(title="Owlix RAG API", version="4.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "Owlix is running",
        "chain_ready": owlix is not None,
        "init_error": init_error,
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if owlix is not None else "degraded",
        "chain_ready": owlix is not None,
        "init_error": init_error,
    }


@app.post("/query")
async def query_endpoint(request: Request):
    if owlix is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Chain not ready",
                "detail": init_error or "Still initialising",
            }
        )
    try:
        body = await request.json()
        query = body.get("query", "")
        session_id = body.get("session_id", "default")
        result = await owlix.run(query, session_id)
        return result
    except Exception as e:
        logger.error("Query failed: %s", traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    logger.info("==> Starting uvicorn on port %d", port)
    uvicorn.run("main:app", host="0.0.0.0", port=port)