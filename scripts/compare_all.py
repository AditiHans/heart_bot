# scripts/compare_all.py
import argparse
import json
from evaluate import load as load_metric
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

MODELS = {
    "taskA": "models/taskA",
    "taskB": "models/taskB",
    "taskB_continual": "models/taskB_continual",
}

def iter_examples(task_path):
    with open(task_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "data" in obj:
        for art in obj["data"]:
            for para in art.get("paragraphs", []):
                ctx = para["context"]
                for qa in para["qas"]:
                    ans = qa.get("answers", [])
                    if isinstance(ans, dict) and "text" in ans and "answer_start" in ans:
                        answers = {"text": list(ans["text"]), "answer_start": list(ans["answer_start"])}
                    else:
                        answers = {"text": [a.get("text","") for a in ans],
                                   "answer_start": [a.get("answer_start",0) for a in ans]}
                    yield {"id": qa["id"], "question": qa["question"], "context": ctx, "answers": answers}
    elif isinstance(obj, list):
        for i, ex in enumerate(obj):
            raw = ex.get("answers", ex.get("answer", ""))
            if isinstance(raw, dict) and "text" in raw and "answer_start" in raw:
                answers = {"text": list(raw["text"]), "answer_start": list(raw["answer_start"])}
            elif isinstance(raw, list):
                answers = {"text": [a.get("text","") for a in raw],
                           "answer_start": [a.get("answer_start",0) for a in raw]}
            else:
                answers = {"text": [str(raw)], "answer_start": [0]}
            yield {"id": ex.get("id", str(i)), "question": ex["question"], "context": ex["context"], "answers": answers}
    else:
        raise ValueError("Unsupported task JSON format.")

def evaluate(model_path, task_path):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    mdl = AutoModelForQuestionAnswering.from_pretrained(model_path)
    qa  = pipeline("question-answering", model=mdl, tokenizer=tok, device=-1)
    metric = load_metric("squad")
    n = 0
    for ex in iter_examples(task_path):
        out = qa(question=ex["question"], context=ex["context"])
        metric.add(
            prediction={"id": ex["id"], "prediction_text": out["answer"]},
            reference={"id": ex["id"], "answers": ex["answers"]},
        )
        n += 1
    res = metric.compute() if n else {"exact_match": 0.0, "f1": 0.0}
    return n, res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_path", required=True)
    args = ap.parse_args()

    print(f"\nComparing models on {args.task_path}\n")
    print("{:<18} {:>8} {:>10} {:>10}".format("Model", "N", "EM", "F1"))
    print("-"*50)
    for name, path in MODELS.items():
        n, res = evaluate(path, args.task_path)
        print("{:<18} {:>8} {:>10.2f} {:>10.2f}".format(name, n, res["exact_match"], res["f1"]))