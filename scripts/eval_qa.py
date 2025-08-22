# scripts/eval_qa.py
import argparse, json, torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from evaluate import load as load_metric

def iter_examples(task_path):
    """
    Yield dicts with: id, question, context, answers (SQuAD format)
    Supports:
      1) SQuAD-like: {"data":[{"paragraphs":[{"context":...,"qas":[...]}]}]}
      2) Flat list: [{"id":..., "question":..., "context":..., "answers": {... or list[...]}}, ...]
    """
    with open(task_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Case 1: SQuAD-like
    if isinstance(obj, dict) and "data" in obj:
        for article in obj["data"]:
            for para in article.get("paragraphs", []):
                context = para["context"]
                for qa in para["qas"]:
                    ans = qa.get("answers", [])
                    # normalize answers into SQuAD dict-of-lists
                    if isinstance(ans, dict) and "text" in ans and "answer_start" in ans:
                        answers = {"text": list(ans["text"]), "answer_start": list(ans["answer_start"])}
                    else:
                        texts = [a.get("text","") for a in ans]
                        starts = [a.get("answer_start", 0) for a in ans]
                        answers = {"text": texts, "answer_start": starts}
                    yield {
                        "id": qa["id"],
                        "question": qa["question"],
                        "context": context,
                        "answers": answers
                    }
        return

    # Case 2: flat list of dicts
    if isinstance(obj, list):
        for i, ex in enumerate(obj):
            context = ex.get("context", "")
            q = ex.get("question", "")
            ex_id = ex.get("id", str(i))
            raw_ans = ex.get("answers", ex.get("answer", ""))

            # normalize answers
            if isinstance(raw_ans, dict) and "text" in raw_ans and "answer_start" in raw_ans:
                answers = {"text": list(raw_ans["text"]), "answer_start": list(raw_ans["answer_start"])}
            elif isinstance(raw_ans, list):
                texts = [a.get("text","") for a in raw_ans]
                starts = [a.get("answer_start", 0) for a in raw_ans]
                answers = {"text": texts, "answer_start": starts}
            else:
                # single string fallback
                answers = {"text": [str(raw_ans)], "answer_start": [0]}
            yield {"id": ex_id, "question": q, "context": context, "answers": answers}
        return

    raise ValueError("Unsupported task JSON format. Expect SQuAD-like or flat list with question/context/answers.")

def evaluate_model(model_path, task_path):
    # device selection (CPU/MPS/GPU)
    device = 0 if torch.cuda.is_available() else -1
    if device == -1 and torch.backends.mps.is_available():
        # transformers pipeline doesn't accept 'mps' index; we'll keep device=-1 and the model will use CPU path
        print("Running on CPU (MPS available but pipeline uses CPU).")

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    mdl = AutoModelForQuestionAnswering.from_pretrained(model_path)
    qa = pipeline("question-answering", model=mdl, tokenizer=tok, device=device)

    metric = load_metric("squad")

    n = 0
    for ex in iter_examples(task_path):
        pred = qa(question=ex["question"], context=ex["context"])
        metric.add(
            prediction={"id": ex["id"], "prediction_text": pred["answer"]},
            reference={"id": ex["id"], "answers": ex["answers"]},
        )
        n += 1

    if n == 0:
        print("No examples found to evaluate.")
        return

    res = metric.compute()
    print(f"\nEvaluated {n} examples from {task_path}")
    print(f"Exact Match (EM): {res['exact_match']:.2f}")
    print(f"F1 Score: {res['f1']:.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task_path", required=True)
    args = ap.parse_args()
    evaluate_model(args.model_path, args.task_path)