from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from . import config
from .rag_core import answer_question

app = FastAPI(title="Heart Health RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS if config.CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    top_k: int = config.TOP_K
    fetch_k: int = config.FETCH_K
    mode: str = config.MODE  # "extractive" or "hybrid"

@app.get("/health")
def health():
    return {"status": "ok", "vector_backend": config.VECTOR_BACKEND, "collection": config.COLLECTION_NAME, "model_path": config.MODEL_PATH}

@app.post("/ask")
def ask(req: AskRequest):
    try:
        if not req.question or not req.question.strip():
            raise HTTPException(status_code=422, detail="Question must be non-empty.")
        result = answer_question(req.question, req.top_k, req.fetch_k, req.mode)
        return {
            "answer": result.get("answer") or "Not enough information in the provided context.",
            "mode_used": result.get("mode_used", req.mode),
            "sources": result.get("sources", []),
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"answer": f"Server error: {e}", "mode_used": "error", "sources": []}