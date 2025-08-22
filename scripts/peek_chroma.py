# scripts/peek_chroma.py
import argparse
from pathlib import Path
import chromadb
from chromadb.config import Settings

def main(persist_dir, collection, n):
    client = chromadb.PersistentClient(
        path=str(Path(persist_dir).resolve()),
        settings=Settings(allow_reset=False)
    )
    col = client.get_collection(collection)
    count = col.count()
    print(f"Collection '{collection}' vectors: {count}")
    if count == 0:
        print("⚠️ Empty collection. Rebuild Chroma."); return

    res = col.query(
        query_texts=["heart disease"],
        n_results=min(n, count),
        include=["documents","metadatas","distances"]
    )
    docs = (res.get("documents") or [[]])[0]
    metas= (res.get("metadatas") or [[]])[0]
    dists= (res.get("distances") or [[]])[0]
    for i, (d, m, dist) in enumerate(zip(docs, metas, dists), 1):
        print(f"\n{i}. source={m.get('source')} dist={float(dist):.4f}\n{d[:300]}...")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist_dir", default="data/chroma")
    ap.add_argument("--collection", default="heart_faq")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    main(args.persist_dir, args.collection, args.n)