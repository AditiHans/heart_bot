from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .rag_core import answer_question


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(title="Heart Health RAG API")


# ─────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        config.CORS_ORIGINS
        if config.CORS_ORIGINS != ["*"]
        else ["*"]
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Static Frontend
# ─────────────────────────────────────────────

# Expected structure:
#
# webapp/
# ├── api.py
# ├── rag_core.py
# ├── config.py
# └── static/
#     ├── index.html
#     ├── style.css
#     └── app.js

STATIC_DIR = Path(__file__).resolve().parent / "static"


# Serve /static/style.css, /static/app.js, etc.
app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


# Serve the frontend at /
@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")


# ─────────────────────────────────────────────
# Request Model
# ─────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    top_k: int = config.TOP_K
    fetch_k: int = config.FETCH_K
    mode: str = config.MODE  # "extractive" or "hybrid"


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_backend": config.VECTOR_BACKEND,
        "collection": config.COLLECTION_NAME,
        "model_path": config.MODEL_PATH,
    }


# Lightweight ping endpoint for keeping an active Render
# instance responsive.
@app.get("/ping")
def ping():
    return {
        "status": "ok"
    }


# ─────────────────────────────────────────────
# Chat / RAG Endpoint
# ─────────────────────────────────────────────

@app.post("/ask")
def ask(req: AskRequest):
    try:
        # Validate question
        if not req.question or not req.question.strip():
            raise HTTPException(
                status_code=422,
                detail="Question must be non-empty.",
            )

        # Run the existing RAG pipeline
        result = answer_question(
            req.question,
            req.top_k,
            req.fetch_k,
            req.mode,
        )

        return {
            "answer": (
                result.get("answer")
                or "Not enough information in the provided context."
            ),
            "mode_used": result.get(
                "mode_used",
                req.mode,
            ),
            "sources": result.get(
                "sources",
                [],
            ),
        }

    except HTTPException:
        raise

    except Exception as e:
        return {
            "answer": f"Server error: {e}",
            "mode_used": "error",
            "sources": [],
        }