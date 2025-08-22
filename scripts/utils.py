import json
from typing import Dict
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_squad(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _normalize_answers(answers_field):
    """
    Normalize answers to HF QA format:
      {"text": [..], "answer_start": [..]}
    Accepts either a list[{"text":..., "answer_start":...}, ...] (SQuAD style)
    or already-normalized dict.
    """
    if isinstance(answers_field, dict) and "text" in answers_field and "answer_start" in answers_field:
        # Already normalized
        return {
            "text": list(answers_field["text"]),
            "answer_start": list(answers_field["answer_start"]),
        }
    # Assume list of dicts
    texts, starts = [], []
    for a in answers_field:
        texts.append(a.get("text", ""))
        starts.append(a.get("answer_start", 0))
    return {"text": texts, "answer_start": starts}

def squad_to_hf(squad: Dict) -> Dataset:
    rows = []
    # defensive checks
    data = squad.get("data", [])
    if not data:
        return Dataset.from_list(rows)
    paragraphs = data[0].get("paragraphs", [])
    for p in paragraphs:
        context = p["context"]
        for qa in p["qas"]:
            rows.append({
                "id": qa["id"],
                "question": qa["question"],
                "context": context,
                "answers": _normalize_answers(qa["answers"])
            })
    return Dataset.from_list(rows)

def make_dd(task_ctx_path: str, split_ratio=0.1) -> DatasetDict:
    squad = load_squad(task_ctx_path)
    ds = squad_to_hf(squad)
    ds = ds.train_test_split(test_size=split_ratio, seed=42)
    return DatasetDict({"train": ds["train"], "validation": ds["test"]})

def qa_preprocess(tokenizer: AutoTokenizer, max_len=384, doc_stride=128):
    def _tokenize(examples):
        questions = [q.strip() for q in examples["question"]]
        tokenized = tokenizer(
            questions,
            examples["context"],
            truncation="only_second",
            max_length=max_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sample_idx = sample_map[i]
            answers = examples["answers"][sample_idx]  # already normalized dict
            # guard for empty answers
            if not answers["answer_start"]:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            sequence_ids = tokenized.sequence_ids(i)

            # find context token span
            idx = 0
            while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if context_start > context_end:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            start_token = None
            end_token = None
            for j in range(context_start, context_end + 1):
                if offsets[j] is None:
                    continue
                # offsets[j] is (char_start, char_end)
                if start_token is None and offsets[j][0] <= start_char < offsets[j][1]:
                    start_token = j
                if offsets[j][0] < end_char <= offsets[j][1]:
                    end_token = j
                    break

            if start_token is None or end_token is None:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                start_positions.append(start_token)
                end_positions.append(end_token)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized
    return _tokenize

def compute_em_f1(preds, refs):
    em = 0
    for p, r in zip(preds, refs):
        em += int(p.strip().lower() == r.strip().lower())
    return {"exact_match": 100.0 * em / len(preds) if preds else 0.0}