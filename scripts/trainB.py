# scripts/trainB.py
import os, inspect
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from utils import make_dd, qa_preprocess, set_seed

BASE = "models/taskA"        # start from Task A weights
OUT_DIR = "models/taskB"     # save Task B fine-tuned model
DATA = "data/taskB_ctx.json" # SQuAD-like JSON for Task B

def supports_arg(cls, arg: str) -> bool:
    try:
        return arg in inspect.signature(cls.__init__).parameters
    except Exception:
        return False

def main():
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(BASE)

    # Build HF DatasetDict from SQuAD-like file
    dd = make_dd(DATA, split_ratio=0.1)
    dd = dd.map(
        qa_preprocess(tokenizer),
        batched=True,
        remove_columns=dd["train"].column_names,
    )

    # Base args most versions understand
    ta_kwargs = dict(
        output_dir=OUT_DIR,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        num_train_epochs=3,
        report_to="none",
        logging_steps=100,
    )

    has_eval_strategy = supports_arg(TrainingArguments, "evaluation_strategy")
    has_save_strategy = supports_arg(TrainingArguments, "save_strategy")
    has_load_best    = supports_arg(TrainingArguments, "load_best_model_at_end")
    has_metric_best  = supports_arg(TrainingArguments, "metric_for_best_model")
    has_fp16         = supports_arg(TrainingArguments, "fp16")

    # Try to align both strategies when supported
    if has_eval_strategy and has_save_strategy:
        ta_kwargs["evaluation_strategy"] = "epoch"
        ta_kwargs["save_strategy"] = "epoch"
        if has_load_best:
            ta_kwargs["load_best_model_at_end"] = True
            if has_metric_best:
                ta_kwargs["metric_for_best_model"] = "eval_loss"
    else:
        # Older versions: no strategy args -> disable best-at-end & use step saving if needed
        if has_load_best:
            ta_kwargs["load_best_model_at_end"] = False
        if not has_save_strategy:
            ta_kwargs["save_steps"] = 500

    # Mixed precision off on CPU/MPS for stability
    if has_fp16:
        ta_kwargs["fp16"] = False

    args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dd["train"],
        eval_dataset=dd["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    os.makedirs(OUT_DIR, exist_ok=True)
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"✅ Saved TaskB model to {OUT_DIR}")

if __name__ == "__main__":
    main()