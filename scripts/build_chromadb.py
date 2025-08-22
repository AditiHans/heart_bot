# scripts/build_chromadb.py
import argparse, json, os
from pathlib import Path
from typing import List, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "heart_faq"

def load_squad_like(path: str) -> List[Tuple[str, str]]:
    """Return list of (doc_id, context_text) from SQuAD-like JSON."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    docs = []
    if isinstance(obj, dict) and "data" in obj:
        k = 0
        for art in obj["data"]:
            for para in art.get("paragraphs", []):
                ctx = (para.get("context") or "").strip()
                if not ctx:
                    continue
                docs.append((f"{Path(path).name}::para::{k}", ctx))
                k += 1
    else:
        for i, row in enumerate(obj if isinstance(obj, list) else []):
            ctx = str(row.get("context", "")).strip()
            if ctx:
                docs.append((f"{Path(path).name}::row::{i}", ctx))
    return docs

def chunk_text(text: str, chunk_chars=1200, overlap=200):
    text = " ".join(text.split())
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def ingest_files(taskA_path, taskB_path, persist_dir, collection,
                 embed_model, reset, batch_size, chunk_chars, overlap):
    persist_dir = str(Path(persist_dir).resolve())
    os.makedirs(persist_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=reset))
    if reset:
        try:
            client.delete_collection(collection)
        except Exception:
            pass
    try:
        col = client.get_collection(collection)
    except Exception:
        col = client.create_collection(collection, metadata={"hnsw:space":"cosine"})

    embedder = SentenceTransformer(embed_model)

    all_docs = []
    for p in [taskA_path, taskB_path]:
        p = str(Path(p).resolve())
        pairs = load_squad_like(p)
        for (doc_id, ctx) in pairs:
            chunks = chunk_text(ctx, chunk_chars=chunk_chars, overlap=overlap)
            for j, ch in enumerate(chunks):
                all_docs.append({
                    "id": f"{doc_id}::chunk::{j}",
                    "text": ch,
                    "meta": {"source": Path(p).name, "parent_id": doc_id, "chunk_id": j}
                })

    if not all_docs:
        raise RuntimeError("No documents found. Ensure *_ctx.json have non-empty 'context'.")

    print(f"Ingesting {len(all_docs)} chunks into '{collection}' at {persist_dir}")
    for i in tqdm(range(0, len(all_docs), batch_size), desc="Add to Chroma"):
        batch = all_docs[i:i+batch_size]
        docs = [b["text"] for b in batch]
        metas = [b["meta"] for b in batch]
        ids   = [b["id"] for b in batch]
        embs  = embedder.encode(docs, convert_to_numpy=True).tolist()
        col.add(documents=docs, metadatas=metas, ids=ids, embeddings=embs)

    print(f"✅ Ingested {len(all_docs)} chunks.")
    print(f"   Collection: {collection}")
    print(f"   Persist dir: {persist_dir}")
    print(f"   Embedder: {embed_model}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskA", required=True)
    ap.add_argument("--taskB", required=True)
    ap.add_argument("--persist_dir", required=True)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--embed_model", default=DEFAULT_EMBED)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--chunk_chars", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=200)
    args = ap.parse_args()

    ingest_files(args.taskA, args.taskB, args.persist_dir, args.collection,
                 args.embed_model, args.reset, args.batch_size,
                 args.chunk_chars, args.overlap)