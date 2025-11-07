# app.py — Heartify RAG API (cloud-only, super light, fixed)
import os, re, asyncio, logging
from typing import List, Optional

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Keep memory small
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CHROMADB_DISABLE_ONNX_RUNTIME", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import httpx
import chromadb

# ---------- Required Cloud Config ----------
CHROMA_HOST     = os.getenv("CHROMA_HOST")      # e.g. "api.trychroma.com"
CHROMA_TENANT   = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_TOKEN    = os.getenv("CHROMA_API_TOKEN")   # header: x-chroma-token

HF_API_TOKEN    = os.getenv("HF_API_TOKEN")
HF_QA_MODEL     = os.getenv("HF_QA_MODEL", "deepset/roberta-base-squad2")

# Retrieval config
COLLECTION      = os.environ.get("COLLECTION", "heart_faq")

# Answering thresholds
TOP_K_DEFAULT   = int(os.environ.get("TOP_K", "4"))
FETCH_K_DEF     = int(os.environ.get("FETCH_K", "12"))   # pull a few more, helps recall
MIN_TOKENS      = int(os.environ.get("MIN_TOKENS", "1")) # allow short spans like "Dyspnea"
MAX_CTX_CHARS   = int(os.environ.get("MAX_CTX_CHARS", "2000"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT_SECS", "25"))

# Optional debug: include top ctx previews + raw HF outputs in response
DEBUG_CTX = os.getenv("DEBUG_CTX", "0") == "1"

# ---------- FastAPI ----------
app = FastAPI(title="Heartify RAG API (Cloud-only, lite)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
log = logging.getLogger("uvicorn.error")

@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

# ---------- Lazy singletons ----------
_collection = None
_http = None

def _missing_env() -> List[str]:
    missing = []
    if not CHROMA_HOST:     missing.append("CHROMA_HOST")
    if not CHROMA_TENANT:   missing.append("CHROMA_TENANT")
    if not CHROMA_DATABASE: missing.append("CHROMA_DATABASE")
    if not CHROMA_TOKEN:    missing.append("CHROMA_API_TOKEN")
    if not HF_API_TOKEN:    missing.append("HF_API_TOKEN")
    return missing

async def _retry(fn, tries=3, delay=1.0):
    last = None
    for _ in range(tries):
        try:
            return await fn()
        except Exception as e:
            last = e
            await asyncio.sleep(delay)
    raise last

async def _get_http():
    global _http
    if _http:
        return _http
    # small timeouts to avoid memory leaks/hangs
    _http = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=10.0))
    return _http

async def _get_collection():
    missing = _missing_env()
    if missing:
        raise HTTPException(status_code=503, detail={"error": "Cloud not configured", "missing_env": missing})

    global _collection
    if _collection is not None:
        return _collection

    def _open_cloud():
        client = chromadb.HttpClient(
            host=CHROMA_HOST,
            ssl=True,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
            headers={"x-chroma-token": CHROMA_TOKEN},
        )
        # IMPORTANT: This collection on Chroma Cloud must have an embedding function configured,
        # so we can query with `query_texts` (not embeddings).
        return client.get_or_create_collection(COLLECTION)

    async def _coro(): return _open_cloud()
    _collection = await _retry(_coro)
    return _collection

# ---------- Hugging Face Inference (QA) ----------
async def _hf_answer(question: str, context: str) -> dict:
    http = await _get_http()
    url = f"https://api-inference.huggingface.co/models/{HF_QA_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": {"question": question, "context": context}}

    for _ in range(3):  # retry a couple times (models may “warm up”)
        r = await http.post(url, headers=headers, json=payload)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, dict) and "answer" in data:
                return {"answer": (data.get("answer") or "").strip(), "score": float(data.get("score") or 0.0)}
            if isinstance(data, str):
                return {"answer": data.strip(), "score": 0.0}
            return {"answer": "", "score": 0.0}
        if r.status_code in (503, 524, 408):
            await asyncio.sleep(1.5)
            continue
        try:
            log.warning("HF error %s: %s", r.status_code, r.json())
        except Exception:
            log.warning("HF error %s: %s", r.status_code, r.text)
        return {"answer": "", "score": 0.0}
    return {"answer": "", "score": 0.0}

# ---------- Helpers ----------
NOISE_RE = re.compile(
    r"\b(dear|hello|hi)\b|thank you|thanks|best regards|regards|yours|dr\.|\bmd\b|"
    r"consult|please rate|assist you further|ask.*(doctor|hcm)|wish you",
    re.I,
)
def clean_context(text: str) -> str:
    t = re.sub(r"(?i)regards.*", "", text)
    t = re.sub(r"(?i)best regards.*", "", t)
    t = re.sub(r"(?i)thank.*", "", t)
    t = re.sub(r"(?i)dr\.\s+[A-Z][a-z].*", "", t)
    return re.sub(r"\s+", " ", t).strip()

def score_doc(doc: str, q_words: set, dist: float, source: str) -> float:
    kw = sum(1 for w in q_words if w in doc.lower())
    bonus = 1.0 if "taska_ctx.json" in (source or "").lower() else 0.0
    penalty = 1.5 if NOISE_RE.search(doc) else 0.0
    return kw + bonus - penalty - float(dist)

async def _retrieve(question: str, fetch_k: int, top_k: int):
    # NOTE: collection must have server-side embedding function set
    col = await _get_collection()
    res = col.query(
        query_texts=[question],                        # << no local embedder!
        n_results=max(fetch_k, top_k),
        include=["documents", "metadatas", "distances"],
    )
    docs  = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    return docs, metas, dists

# ---------- Schemas ----------
class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: Optional[int] = Field(default=TOP_K_DEFAULT, ge=1, le=10)
    fetch_k: Optional[int] = Field(default=FETCH_K_DEF, ge=1, le=50)
    extra_contexts: Optional[List[str]] = None

# ---------- Endpoints ----------
@app.get("/health")
async def health():
    missing = _missing_env()
    ok = len(missing) == 0
    return {
        "ok": ok,
        "missing_env": missing,
        "collection": COLLECTION,
        "hf_model": HF_QA_MODEL,
    }

@app.get("/ping")
async def ping():
    return {"pong": True}

@app.post("/ask")
async def ask(req: AskRequest):
    async def _work():
        q = req.question.strip()
        if not q:
            raise HTTPException(status_code=422, detail="Question must be non-empty.")

        docs, metas, dists = await _retrieve(q, req.fetch_k or FETCH_K_DEF, req.top_k or TOP_K_DEFAULT)

        # Add ephemeral user contexts (not persisted)
        extra_docs = req.extra_contexts or []
        docs  += extra_docs
        metas += [{"source": "ephemeral"} for _ in extra_docs]
        dists += [0.25 for _ in extra_docs]

        if not docs:
            return {"answer": "Not enough information in the provided context.", "confidence": 0.0, "sources": []}

        q_words = {w for w in re.findall(r"[A-Za-z]+", q.lower()) if len(w) > 3}
        scored = []
        for d, m, dist in zip(docs, metas, dists):
            d_clean = clean_context(d or "")
            if len(d_clean) < 30:          # <-- lowered cutoff (was 60)
                continue
            sc = score_doc(d_clean, q_words, dist, (m or {}).get("source", ""))
            scored.append((sc, d_clean, (m or {}).get("source", "unknown"), float(dist)))

        scored.sort(key=lambda x: x[0], reverse=True)
        keep = scored[: max(req.top_k or TOP_K_DEFAULT, 1)]
        if not keep:
            return {"answer": "Not enough information in the provided context.", "confidence": 0.0, "sources": []}

        # Ask HF for each top doc; accept any non-empty answer (do not drop short spans)
        best = {"text": "", "score": -1.0, "len": 0}
        raw_attempts = []
        for _, d_clean, _, _ in keep:
            ctx = d_clean[:MAX_CTX_CHARS]
            qa_out = await _hf_answer(q, ctx)
            ans = re.sub(r"\s+", " ", (qa_out.get("answer") or "")).strip()
            score = float(qa_out.get("score") or 0.0)
            raw_attempts.append({"ctx_preview": ctx[:300], "answer": ans, "score": score})
            if ans:  # accept short answers too
                tok_len = len(ans.split())
                if (score > best["score"]) or (abs(score - best["score"]) < 1e-6 and tok_len > best["len"]):
                    best = {"text": ans, "score": score, "len": tok_len}

        # Friendly final fallback: first sentence of best context
        if not best["text"]:
            top_ctx = keep[0][1]
            first_sentence = re.split(r'(?<=[.!?])\s+', top_ctx.strip())[0]
            best = {"text": first_sentence, "score": 0.0, "len": len(first_sentence.split())}

        sources = [{"source": s, "distance": round(d, 4)} for _, _, s, d in keep]
        resp = {
            "answer": best["text"],
            "confidence": round(max(best["score"], 0.0), 4),
            "sources": sources,
        }
        if DEBUG_CTX:
            resp["debug"] = raw_attempts[:2]
        return resp

    try:
        return await asyncio.wait_for(_work(), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out, try narrowing your question.")
