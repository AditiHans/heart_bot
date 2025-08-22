import re, torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, pipeline
from . import config
from .vector_store import query as vs_query

device = 0 if torch.cuda.is_available() else -1  # Render Free CPU → -1

# Extractive QA model (your continual model repo or public HF)
_tok_ex = AutoTokenizer.from_pretrained(config.MODEL_PATH, use_fast=True)
_mod_ex = AutoModelForQuestionAnswering.from_pretrained(config.MODEL_PATH)
_qa_ex = pipeline("question-answering", model=_mod_ex, tokenizer=_tok_ex, device=device,
                  handle_impossible_answer=True, max_answer_len=64, align_to_words=True)

# Lazy generative (optional)
_gen_pipe = None
def _get_gen():
    global _gen_pipe
    if _gen_pipe is None:
        tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
        mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        _gen_pipe = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=device)
    return _gen_pipe

NOISE_RE = re.compile(r"(dear|hello|thank|regards|dr\.|please rate|assist you further)", re.I)

def _is_noisy(t: str) -> bool:
    return bool(NOISE_RE.search((t or "")))

def _clean(t: str) -> str:
    t = re.sub(r"(?i)regards.*", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _score(q: str, doc: str, meta: dict, dist: float):
    q_words = {w for w in re.findall(r"[A-Za-z]+", q.lower()) if len(w) > 3}
    kw = sum(1 for w in q_words if w in (doc or "").lower())
    src_bonus = 1.0 if isinstance(meta, dict) and "taskA_ctx.json" in (meta.get("source") or "") else 0.0
    penalty = 1.5 if _is_noisy(doc) else 0.0
    return kw + src_bonus - penalty - float(dist or 0.0)

def retrieve(question: str, fetch_k: int, top_k: int):
    docs, metas, dists = vs_query(question, top_k=max(fetch_k,1))
    scored = []
    for d,m,dist in zip(docs, metas, dists):
        s = _score(question, d, m or {}, dist)
        d2 = _clean(d or "")
        if len(d2) >= 60:
            scored.append((s, d2, m or {}, float(dist)))
    scored.sort(key=lambda x: x[0], reverse=True)
    keep = scored[:max(top_k,1)]
    return [k[1] for k in keep], [k[2] for k in keep], [k[3] for k in keep]

def answer_question(question: str, top_k: int, fetch_k: int, mode: str):
    docs, metas, dists = retrieve(question, fetch_k, top_k)
    if not docs:
        return {"answer": "Not enough information in the provided context.", "mode_used": "none", "sources": []}

    sources = []
    for d, m, dist in zip(docs, metas, dists):
        sources.append({
            "source": (m.get("source") if isinstance(m, dict) else "unknown") or "unknown",
            "distance": round(float(dist), 4)
        })

    # Extractive first
    try:
        ext = _qa_ex({"context": docs[0], "question": question})
        span = re.sub(r"\s+", " ", (ext.get("answer") or "")).strip()
        conf = float(ext.get("score") or 0.0)
    except Exception as e:
        return {"answer": f"Extraction error: {e}", "mode_used": "extractive-error", "sources": sources}

    if mode == "extractive":
        return {"answer": span or "Not enough information in the provided context.", "mode_used": "extractive", "sources": sources}

    if mode == "hybrid":
        if len(span.split()) >= config.MIN_TOKENS and conf >= config.MIN_CONF:
            return {"answer": span, "mode_used": "hybrid(extractive)", "sources": sources}
        # try tiny T5 if available
        try:
            gen = _get_gen()
            merged = " ".join(docs)[:1400]
            prompt = (
                "Answer ONLY from the context.\n"
                "Short, factual. If missing: 'Not enough information in the provided context.'\n\n"
                f"Question: {question}\nContext:\n{merged}"
            )
            out = gen(prompt, max_length=220, do_sample=False)[0]["generated_text"]
            return {"answer": re.sub(r'\s+', ' ', out).strip(), "mode_used": "hybrid(gen)", "sources": sources}
        except Exception as e:
            return {"answer": span or "Not enough information in the provided context.",
                    "mode_used": f"hybrid(extractive-fallback; gen_error={e})", "sources": sources}

    # pure gen (optional)
    try:
        gen = _get_gen()
        merged = " ".join(docs)[:1400]
        prompt = (
            "Answer ONLY from the context.\n"
            "Short, factual. If missing: 'Not enough information in the provided context.'\n\n"
            f"Question: {question}\nContext:\n{merged}"
        )
        out = gen(prompt, max_length=220, do_sample=False)[0]["generated_text"]
        return {"answer": re.sub(r'\s+', ' ', out).strip(), "mode_used": "generative", "sources": sources}
    except Exception as e:
        return {"answer": f"Generative error: {e}", "mode_used": "generative-error", "sources": sources}