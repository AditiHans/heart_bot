# scripts/validate_ctx.py
import json, sys
from pathlib import Path

def validate(path):
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)

    total = 0
    empty = 0
    short = 0

    if not isinstance(obj, dict) or "data" not in obj:
        print(f"{p.name}: Not SQuAD-like (missing 'data')."); sys.exit(1)

    for art in obj["data"]:
        for para in art.get("paragraphs", []):
            total += 1
            ctx = (para.get("context") or "").strip()
            if not ctx:
                empty += 1
            elif len(ctx) < 80:
                short += 1

    print(f"{p.name}: paragraphs={total}, empty={empty}, short(<80ch)={short}")
    if empty or short:
        print("⚠️ Consider regenerating contexts or increasing chunk size.")
    else:
        print("✅ Contexts look OK.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_ctx.py data/taskA_ctx.json"); sys.exit(1)
    validate(sys.argv[1])