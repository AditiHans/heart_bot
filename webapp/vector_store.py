import os
from sentence_transformers import SentenceTransformer

BACKEND = os.getenv("VECTOR_BACKEND", "QDRANT").upper()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "heart_faq")

_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if BACKEND == "QDRANT":
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    import numpy as np

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    def ensure_collection(dim=384):
        try:
            collections = [c.name for c in client.get_collections().collections]
        except Exception:
            collections = []
        if COLLECTION_NAME not in collections:
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    ensure_collection()

    def upsert_texts(ids, texts, metas):
        vectors = _embedder.encode(texts, convert_to_numpy=True).astype(np.float32)
        points = [PointStruct(id=ids[i], vector=vectors[i].tolist(), payload=metas[i]) for i in range(len(ids))]
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    def query(question, top_k=5):
        vec = _embedder.encode([question], convert_to_numpy=True)[0].astype("float32")
        hits = client.search(collection_name=COLLECTION_NAME, query_vector=vec.tolist(), limit=top_k, with_payload=True)
        docs, metas, dists = [], [], []
        for h in hits:
            payload = h.payload or {}
            docs.append(payload.get("text", ""))
            metas.append(payload)
            # qdrant score is similarity; we convert to a distance-like value
            dists.append(1 - float(h.score or 0.0))
        return docs, metas, dists

else:
    # fallback (not recommended on Render Free): local chroma
    import chromadb
    from chromadb.config import Settings
    PERSIST_DIR = os.getenv("PERSIST_DIR", "./data/chroma")
    _chroma = chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings(allow_reset=False))
    try:
        _col = _chroma.get_collection(COLLECTION_NAME)
    except Exception:
        _col = _chroma.create_collection(COLLECTION_NAME)

    def upsert_texts(ids, texts, metas):
        _col.add(ids=ids, documents=texts, metadatas=metas)

    def query(question, top_k=5):
        vec = _embedder.encode([question], convert_to_numpy=True).tolist()
        res = _col.query(query_embeddings=vec, n_results=top_k, include=["documents","metadatas","distances"])
        docs  = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return docs, metas, dists