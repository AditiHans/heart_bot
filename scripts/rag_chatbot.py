# scripts/rag_chatbot.py
import argparse, re
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    pipeline,
)

COLLECTION = "heart_faq"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEN_MODEL = "google/flan-t5-base"

# Heuristics to detect conversational/MedDialog-y noise
NOISE_PATTERNS = [
    r"\bdear\b", r"\bhello\b", r"\bhi\b", r"thank you", r"thanks",
    r"best regards", r"regards", r"yours", r"dr\.", r"\bmd\b",
    r"consult", r"please rate", r"hope i", r"assist you further",
    r"ask.*(doctor|hcm)", r"thank.*(query|asking)", r"wish you"
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS), flags=re.I)

def is_noisy(text: str) -> bool:
    t = text.lower()
    # conversational markers or explicit “doctor” persona
    if NOISE_RE.search(t):
        return True
    if re.search(r"\bi\b.*doctor", t):
        return True
    return False

def clean_context(text: str) -> str:
    # Remove greetings/sign-offs/doctor signatures + collapse whitespace
    txt = re.sub(r"(?i)regards.*", "", text)
    txt = re.sub(r"(?i)best regards.*", "", txt)
    txt = re.sub(r"(?i)thank.*", "", txt)
    txt = re.sub(r"(?i)dr\.\s+[A-Z][a-z].*", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def keyword_score(text, q_words):
    t = text.lower()
    return sum(1 for w in q_words if w in t)

def source_bonus(meta):
    # Prefer your factual FAQ source
    src = (meta.get("source") or "").lower()
    bonus = 0.0
    if "taska_ctx.json" in src:
        bonus += 1.0   # BHF/NHLBI FAQ
    if "taskb_ctx.json" in src:
        bonus += 0.0
    return bonus

def load_retriever(persist_dir):
    client = chromadb.PersistentClient(
        path=str(Path(persist_dir).resolve()),
        settings=Settings(allow_reset=False)
    )
    col = client.get_collection(COLLECTION)
    embedder = SentenceTransformer(EMBED_MODEL)
    return col, embedder

def load_extractive(model_path):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    mdl = AutoModelForQuestionAnswering.from_pretrained(model_path)
    qa = pipeline(
        "question-answering",
        model=mdl,
        tokenizer=tok,
        device=-1,
        handle_impossible_answer=True,
        max_answer_len=64,
        align_to_words=True
    )
    return qa

def load_generative(gen_model_name):
    tok = AutoTokenizer.from_pretrained(gen_model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
    gen = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=-1)
    return gen

def main(persist_dir, model_path, mode, gen_model, top_k, show_ctx, ctx_chars,
         min_tokens, min_conf, fetch_k, max_ctx_chars):
    # retriever
    col, embedder = load_retriever(persist_dir)

    # answerers
    qa_extractive = load_extractive(model_path)
    qa_generative = None
    if mode in ("generative", "hybrid"):
        qa_generative = load_generative(gen_model or DEFAULT_GEN_MODEL)

    print(f"\n💬 Heart Health Chatbot (mode: {mode}). Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Goodbye 👋"); break

        # Retrieve with embeddings
        q_emb = embedder.encode([q], convert_to_numpy=True).tolist()
        res = col.query(
            query_embeddings=q_emb,
            n_results=max(fetch_k, top_k),           # pull more, then filter/rerank
            include=["documents", "metadatas", "distances"]
        )

        docs  = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        if not docs:
            print("⚠️ No documents retrieved. Rebuild index or try a broader question.")
            continue

        # Build question keywords
        q_words = {w for w in re.findall(r"[A-Za-z]+", q.lower()) if len(w) > 3}

        # Score: keywords + source bonus - noise penalty - distance
        scored = []
        for d, m, dist in zip(docs, metas, dists):
            kw = keyword_score(d, q_words)
            bon = source_bonus(m)
            noisy = is_noisy(d)
            score = kw + bon - (1.5 if noisy else 0.0) - float(dist)
            scored.append((score, d, m, dist, noisy))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Keep top_k after cleaning & length check
        keep_docs, keep_metas, keep_dists, keep_scores = [], [], [], []
        for s, d, m, dist, noisy in scored:
            d_clean = clean_context(d)
            if len(d_clean) < 60:
                continue
            keep_docs.append(d_clean)
            keep_metas.append(m)
            keep_dists.append(float(dist))
            keep_scores.append(float(s))
            if len(keep_docs) >= top_k:
                break

        if not keep_docs:
            print("⚠️ Retrieved only noisy/short documents. Re-ingest with larger chunk size.")
            continue

        # print("\nTop sources (post-filter):")
        # for i, (m, d, s) in enumerate(zip(keep_metas, keep_dists, keep_scores), 1):
        #     print(f"  {i}. {m.get('source','unknown')} (dist={d:.4f}, score={s:.2f})")

        merged_context = "\n\n".join(keep_docs)
        # Truncate long context for generative focus
        if len(merged_context) > max_ctx_chars:
            merged_context = merged_context[:max_ctx_chars]

        # if show_ctx:
        #     print("\n[Context preview]")
        #     preview = merged_context[:ctx_chars]
        #     print(preview + ("..." if len(merged_context) > ctx_chars else ""))

        # ----- Answering -----
        answer_text = ""
        conf_score  = 0.0

        if mode in ("extractive", "hybrid"):
            # Ask each doc separately, pick longest high-confidence span
            best = {"text": "", "score": -1.0, "len": 0}
            for d in keep_docs:
                out = qa_extractive({"question": q, "context": d})
                ans = re.sub(r"\s+", " ", out.get("answer", "")).strip()
                score = float(out.get("score", 0.0))
                tok_len = len(ans.split())
                if tok_len < min_tokens:
                    continue
                if (score > best["score"]) or (abs(score - best["score"]) < 1e-6 and tok_len > best["len"]):
                    best = {"text": ans, "score": score, "len": tok_len}

            answer_text = best["text"]
            conf_score  = best["score"]

            if mode == "extractive":
                print("\n🤖 Answer:", answer_text if answer_text else "(no confident span found)")
                print("-"*60)
                continue

        # Hybrid: fallback to generative if extractive is short/low-conf
        needs_fallback = (len(answer_text.split()) < min_tokens) or (conf_score < min_conf)
        if mode == "hybrid" and needs_fallback:
            if qa_generative is None:
                qa_generative = load_generative(gen_model or DEFAULT_GEN_MODEL)

            prompt = (
                "You are a concise, factual medical assistant.\n"
                "Answer ONLY from the provided context.\n"
                "Rules:\n"
                "- Use short sentences or bullet points.\n"
                "- Do NOT include greetings, sign-offs, or doctor names.\n"
                "- Do NOT invent details.\n"
                "- If not in context, reply: 'Not enough information in the provided context.'\n\n"
                f"Question: {q}\n"
                f"Context:\n{merged_context}"
            )
            out = qa_generative(prompt, max_length=220, do_sample=False)[0]["generated_text"]
            answer_text = re.sub(r"\s+", " ", out).strip()

        if mode == "generative":
            if qa_generative is None:
                qa_generative = load_generative(gen_model or DEFAULT_GEN_MODEL)
            prompt = (
                "You are a concise, factual medical assistant.\n"
                "Answer ONLY from the provided context.\n"
                "Rules:\n"
                "- Use short sentences or bullet points.\n"
                "- Do NOT include greetings, sign-offs, or doctor names.\n"
                "- Do NOT invent details.\n"
                "- If not in context, reply: 'Not enough information in the provided context.'\n\n"
                f"Question: {q}\n"
                f"Context:\n{merged_context}"
            )
            out = qa_generative(prompt, max_length=220, do_sample=False)[0]["generated_text"]
            answer_text = re.sub(r"\s+", " ", out).strip()

        print(f"\n🤖 Answer: {answer_text}")
        print("-"*60)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist_dir", default="data/chroma")
    ap.add_argument("--model_path",  default="models/taskB_continual",
                    help="Used in extractive/hybrid modes")
    ap.add_argument("--mode", choices=["extractive", "generative", "hybrid"], default="hybrid")
    ap.add_argument("--gen_model", default=None, help="e.g., google/flan-t5-base")
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--fetch_k", type=int, default=10, help="retrieve this many, then filter to top_k")
    ap.add_argument("--show_ctx", action="store_true")
    ap.add_argument("--ctx_chars", type=int, default=900)
    ap.add_argument("--max_ctx_chars", type=int, default=1400, help="truncate merged context to this length")
    ap.add_argument("--min_tokens", type=int, default=6, help="min tokens for extractive span")
    ap.add_argument("--min_conf", type=float, default=0.25, help="min confidence for extractive span")
    args = ap.parse_args()

    main(
        args.persist_dir, args.model_path, args.mode, args.gen_model,
        args.top_k, args.show_ctx, args.ctx_chars, args.min_tokens, args.min_conf,
        args.fetch_k, args.max_ctx_chars
    )