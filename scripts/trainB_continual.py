# scripts/trainB_continual.py
import os, inspect, random
from datasets import concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from utils import make_dd, qa_preprocess, set_seed

BASE = "models/taskA"              # start from Task A weights
OUT_DIR = "models/taskB_continual" # save continual model here
TASKA = "data/taskA_ctx.json"
TASKB = "data/taskB_ctx.json"

# How much Task A to replay during Task B training
REPLAY_RATIO = 0.3
SEED = 42

def supports_arg(cls, arg):
    try:
        return arg in inspect.signature(cls.__init__).parameters
    except Exception:
        return False

def main():
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(BASE)

    # Build Task B (target) dataset
    ddB = make_dd(TASKB, split_ratio=0.1)
    # Build Task A (replay source) dataset
    ddA = make_dd(TASKA, split_ratio=0.1)

    # Tokenizer-driven preprocessing
    proc = qa_preprocess(tokenizer)

    # Tokenize Task B train/val
    trainB_tok = ddB["train"].map(proc, batched=True, remove_columns=ddB["train"].column_names)
    valB_tok   = ddB["validation"].map(proc, batched=True, remove_columns=ddB["validation"].column_names)

    # Build replay buffer from Task A train
    trainA = ddA["train"]
    nA = len(trainA)
    k  = max(1, int(len(ddB["train"]) * REPLAY_RATIO))
    idxs = list(range(nA))
    random.Random(SEED).shuffle(idxs)
    idxs = idxs[:min(k, nA)]
    bufferA = trainA.select(idxs)

    # Tokenize replay buffer
    bufferA_tok = bufferA.map(proc, batched=True, remove_columns=bufferA.column_names)

    # Merge Task B with replay buffer
    train_concat = concatenate_datasets([trainB_tok, bufferA_tok])

    # Version-safe training args
    ta_kwargs = dict(
        output_dir=OUT_DIR,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        num_train_epochs=3,
        report_to="none",
        logging_steps=100,
        seed=SEED,
        dataloader_num_workers=0,  # stable on macOS
    )

    has_eval_strategy = supports_arg(TrainingArguments, "evaluation_strategy")
    has_save_strategy = supports_arg(TrainingArguments, "save_strategy")
    has_load_best     = supports_arg(TrainingArguments, "load_best_model_at_end")
    has_metric_best   = supports_arg(TrainingArguments, "metric_for_best_model")
    has_fp16          = supports_arg(TrainingArguments, "fp16")

    if has_eval_strategy and has_save_strategy:
        ta_kwargs["evaluation_strategy"] = "epoch"
        ta_kwargs["save_strategy"] = "epoch"
        if has_load_best:
            ta_kwargs["load_best_model_at_end"] = True
            if has_metric_best:
                ta_kwargs["metric_for_best_model"] = "eval_loss"
    else:
        if has_load_best:
            ta_kwargs["load_best_model_at_end"] = False
        if not has_save_strategy:
            ta_kwargs["save_steps"] = 500

    if has_fp16:
        ta_kwargs["fp16"] = False  # CPU/MPS-safe

    args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_concat,
        eval_dataset=valB_tok,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    os.makedirs(OUT_DIR, exist_ok=True)
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f" Saved continual model with replay to {OUT_DIR}")

if __name__ == "__main__":
    main()