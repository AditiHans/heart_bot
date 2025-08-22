# scripts/eval_qa_soft.py
import argparse, json, re, string, torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from evaluate import load as load_metric

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return s

def iter_examples(task_path):
    with open(task_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "data" in obj:
        for art in obj["data"]:
            for para in art.get("paragraphs", []):
                ctx = para["context"]
                for qa in para["qas"]:
                    ans = qa.get("answers", [])
                    if isinstance(ans, dict):
                        texts  = [normalize_text(t) for t in ans.get("text", [])]
                        starts = ans.get("answer_start", [0]*len(texts))
                    else:
                        texts  = [normalize_text(a.get("text","")) for a in ans]
                        starts = [a.get("answer_start", 0) for a in ans]
                    yield {
                        "id": qa["id"],
                        "question": qa["question"],
                        "context": ctx,
                        "answers": {"text": texts, "answer_start": starts},
                    }
    elif isinstance(obj, list):
        for i, ex in enumerate(obj):
            ctx = ex["context"]; q = ex["question"]; exid = ex.get("id", str(i))
            raw = ex.get("answers", ex.get("answer", ""))
            if isinstance(raw, dict):
                texts  = [normalize_text(t) for t in raw.get("text", [])]
                starts = raw.get("answer_start", [0]*len(texts))
            elif isinstance(raw, list):
                texts  = [normalize_text(a.get("text","")) for a in raw]
                starts = [a.get("answer_start", 0) for a in raw]
            else:
                texts, starts = [normalize_text(str(raw))], [0]
            yield {"id": exid, "question": q, "context": ctx, "answers": {"text":texts, "answer_start":starts}}
    else:
        raise ValueError("Unsupported task JSON format.")

def evaluate_model(model_path, task_path):
    device = 0 if torch.cuda.is_available() else -1
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    mdl = AutoModelForQuestionAnswering.from_pretrained(model_path)
    qa  = pipeline("question-answering", model=mdl, tokenizer=tok, device=device)
    metric = load_metric("squad")

    n = 0
    for ex in iter_examples(task_path):
        pred = qa(question=ex["question"], context=ex["context"])
        pred_norm = normalize_text(pred["answer"])
        metric.add(
            prediction={"id": ex["id"], "prediction_text": pred_norm},
            reference={"id": ex["id"], "answers": {"text": ex["answers"]["text"], "answer_start": ex["answers"]["answer_start"]}},
        )
        n += 1
    res = metric.compute() if n else {"exact_match": 0.0, "f1": 0.0}
    print(f"\nEvaluated {n} examples from {task_path}")
    print(f"Exact Match (EM): {res['exact_match']:.2f}")
    print(f"F1 Score: {res['f1']:.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task_path", required=True)
    args = ap.parse_args()
    evaluate_model(args.model_path, args.task_path)