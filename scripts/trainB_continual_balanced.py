# scripts/trainB_continual_balanced.py
import os, inspect, random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from utils import make_dd, qa_preprocess, set_seed

BASE    = "models/taskA"              # start from Task A weights
OUT_DIR = "models/taskB_continual"    # save continual model here
TASKA   = "data/taskA_ctx.json"
TASKB   = "data/taskB_ctx.json"

REPLAY_RATIO = 0.5        # aim for 50/50 in each batch
LR           = 1e-5       # smaller LR for stability during replay
UNFREEZE_TOP = 2          # unfreeze only top N encoder layers (BERT)
BATCH_SIZE   = 12         # use an even number for clean 50/50 batches
SEED         = 42

def supports_arg(cls, arg: str) -> bool:
    try:
        return arg in inspect.signature(cls.__init__).parameters
    except Exception:
        return False

def freeze_lower_bert_layers(model, keep_top: int = 2):
    """
    Freeze all encoder layers except the top `keep_top` layers.
    Always keep the QA head trainable.
    """
    enc_layers = None
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        enc_layers = model.bert.encoder.layer
    elif hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        enc_layers = model.roberta.encoder.layer

    if enc_layers is not None:
        n = len(enc_layers)
        for layer in enc_layers[: max(0, n - keep_top)]:
            for p in layer.parameters():
                p.requires_grad = False

    # keep QA head trainable
    if hasattr(model, "qa_outputs"):
        for p in model.qa_outputs.parameters():
            p.requires_grad = True

class BalancedBatchDataset(Dataset):
    """
    A length-known dataset that interleaves A and B examples.
    The order list alternates A,B,A,B,... so a DataLoader with an even batch_size
    will naturally produce ~50/50 batches.
    """
    def __init__(self, dsA: HFDataset, dsB: HFDataset, seed: int = 42):
        self.dsA = dsA
        self.dsB = dsB
        self.seed = seed

        rng = random.Random(seed)
        self.idxA = list(range(len(self.dsA)))
        self.idxB = list(range(len(self.dsB)))
        rng.shuffle(self.idxA)
        rng.shuffle(self.idxB)

        # Build an interleaved index plan: [("A", i), ("B", j), ...]
        i = j = 0
        order = []
        while i < len(self.idxA) or j < len(self.idxB):
            if i < len(self.idxA):
                order.append(("A", self.idxA[i])); i += 1
            if j < len(self.idxB):
                order.append(("B", self.idxB[j])); j += 1
        self.order = order

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        src, i = self.order[idx]
        if src == "A":
            return self.dsA[i]
        else:
            return self.dsB[i]

class BalancedReplayTrainer(Trainer):
    """
    Custom Trainer that feeds a balanced A/B dataset to ensure 50/50 mini-batches.
    """
    def __init__(self, *args, balanced_train_ds: Dataset, **kwargs):
        super().__init__(*args, **kwargs)
        self._balanced_train_ds = balanced_train_ds

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._balanced_train_ds,
            batch_size=self.args.per_device_train_batch_size,
            num_workers=0,
            collate_fn=self.data_collator,
            shuffle=False,      # order is predetermined for balance
            drop_last=False,
        )

def main():
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    model     = AutoModelForQuestionAnswering.from_pretrained(BASE)

    # Freeze lower layers to reduce interference + compute
    freeze_lower_bert_layers(model, keep_top=UNFREEZE_TOP)

    # Build Task B (target) and Task A (replay source)
    ddB = make_dd(TASKB, split_ratio=0.1)
    ddA = make_dd(TASKA, split_ratio=0.1)

    proc = qa_preprocess(tokenizer)

    # Tokenize
    trainB_tok = ddB["train"].map(proc, batched=True, remove_columns=ddB["train"].column_names)
    valB_tok   = ddB["validation"].map(proc, batched=True, remove_columns=ddB["validation"].column_names)
    trainA_tok = ddA["train"].map(proc, batched=True, remove_columns=ddA["train"].column_names)

    # Size-match A replay to B according to REPLAY_RATIO
    target_A = max(1, int(len(trainB_tok) * REPLAY_RATIO))
    idxs = list(range(len(trainA_tok)))
    random.Random(SEED).shuffle(idxs)
    idxs = idxs[: min(target_A, len(trainA_tok))]
    bufferA_tok = trainA_tok.select(idxs)

    # Build the balanced dataset used for training
    balanced_train_ds = BalancedBatchDataset(bufferA_tok, trainB_tok, seed=SEED)

    # ---- Training args (version-safe) ----
    ta_kwargs = dict(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,    # even number, e.g., 12
        per_device_eval_batch_size=16,
        learning_rate=LR,
        num_train_epochs=3,
        report_to="none",
        logging_steps=100,
        dataloader_num_workers=0,
        seed=SEED,
    )
    has_eval_strategy = supports_arg(TrainingArguments, "evaluation_strategy")
    has_save_strategy = supports_arg(TrainingArguments, "save_strategy")
    has_load_best     = supports_arg(TrainingArguments, "load_best_model_at_end")
    has_metric_best   = supports_arg(TrainingArguments, "metric_for_best_model")
    has_fp16          = supports_arg(TrainingArguments, "fp16")

    if has_eval_strategy and has_save_strategy:
        ta_kwargs["evaluation_strategy"] = "epoch"
        ta_kwargs["save_strategy"]       = "epoch"
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
        ta_kwargs["fp16"] = False  # CPU/MPS safe

    args = TrainingArguments(**ta_kwargs)

    trainer = BalancedReplayTrainer(
        model=model,
        args=args,
        balanced_train_ds=balanced_train_ds,
        eval_dataset=valB_tok,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        train_dataset=trainB_tok,   # required by API but unused (we override dataloader)
    )

    trainer.train()
    os.makedirs(OUT_DIR, exist_ok=True)
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f" Saved balanced-replay model to {OUT_DIR}")

if __name__ == "__main__":
    main()