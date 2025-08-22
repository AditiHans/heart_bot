import json
import argparse
from pathlib import Path

def convert(in_path: Path, out_path: Path):
    with open(in_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Expecting list of {"question": "...", "answer": "..."}
    squad_like = {"data": [{"title": in_path.stem, "paragraphs": []}]}

    for i, ex in enumerate(raw):
        q = ex.get("question", "").strip()
        a = ex.get("answer", "").strip()
        if not q or not a:
            continue

        context = a  # simplest: answer is the context so span is valid
        answer_start = context.find(a)
        if answer_start == -1:
            # fallback: ensure answer inside context
            context = a
            answer_start = 0

        qa = {
            "qas": [{
                "id": f"{in_path.stem}-{i}",
                "question": q,
                "answers": [{"text": a, "answer_start": answer_start}],
                "is_impossible": False
            }],
            "context": context
        }
        squad_like["data"][0]["paragraphs"].append(qa)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(squad_like, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="path to taskX.json")
    ap.add_argument("--out", dest="out", required=True, help="path to output taskX_ctx.json")
    args = ap.parse_args()

    convert(Path(args.inp), Path(args.out))
    print(f"Saved: {args.out}")