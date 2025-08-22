import os, json, uuid
from webapp.vector_store import upsert_texts

def load_ctx(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    seen, ids, texts, metas = set(), [], [], []
    for item in data:
        ctx = (item.get("context") or "").strip()
        if not ctx or ctx in seen: continue
        seen.add(ctx)
        ids.append(str(uuid.uuid4()))
        texts.append(ctx)
        metas.append({"source": os.path.basename(path), "text": ctx})
    return ids, texts, metas

def push(path):
    ids, texts, metas = load_ctx(path)
    B=512
    for i in range(0, len(ids), B):
        upsert_texts(ids[i:i+B], texts[i:i+B], metas[i:i+B])
        print(f"Pushed {min(i+B, len(ids))}/{len(ids)} from {os.path.basename(path)}")

if __name__ == "__main__":
    for p in ("data/taskA_ctx.json", "data/taskB_ctx.json"):
        if os.path.exists(p):
            push(p)