from app.utils import preprocess
import urllib
import csv
import os
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import load_from_disk


# --- Device detection ---
if torch.cuda.is_available():
    device = "cuda"
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
elif torch.backends.mps.is_available():
    device = "mps"
    use_bf16 = False
    use_fp16 = False
else:
    device = "cpu"
    use_bf16 = False
    use_fp16 = False
    
if device == "cuda" and use_bf16:
    load_dtype = torch.bfloat16
elif device == "cuda" and use_fp16:
    load_dtype = torch.float16
else:
    load_dtype = torch.float32  # MPS/CPU -> fp32

import evaluate


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"

# download label mapping
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]



# --- Tokenizer: keep short max_length to save memory ---
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, model_max_length=128)


def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
        padding=False  # we will pad per-batch via DataCollatorWithPadding
    )

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8 if (device == "cuda" and (use_bf16 or use_fp16)) else None
)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=3, torch_dtype=load_dtype
)

model.gradient_checkpointing_enable()
model.config.use_cache = False

#### DATASET LOADING


dataset_path = "data/dataset"  # same path you used before
dataset = load_from_disk(dataset_path)


# ---- COPY-PASTE FROM HERE ----
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding

def make_trainer_ready(
    raw_ds: DatasetDict,
    model_name: str = "cardiffnlp/twitter-roberta-base-sep2022",
    train_frac: float = 0.2,
    val_frac: float = 0.2,
    seed: int = 42,
    label_col: str = "label",
    text_col: str = "text",
    max_length: int = 128,
    pad_to_multiple_of_8_on_cuda: bool = True,
):
    """
    Returns (train_ds, eval_ds, data_collator, tokenizer) ready for HF Trainer.
    - Ensures there's a validation split (creates one from train if missing).
    - Takes fractional subsets, stratified by label when possible.
    - Tokenizes and keeps only the columns Trainer expects.
    """
    assert 0 < train_frac <= 1.0, "train_frac must be in (0,1]."
    assert 0 < val_frac <= 1.0, "val_frac must be in (0,1]."
    assert text_col in raw_ds["train"].column_names, f"Missing text column: {text_col}"
    assert label_col in raw_ds["train"].column_names, f"Missing label column: {label_col}"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=max_length)

    # 1) Ensure we have a validation split
    if "validation" not in raw_ds:
        split = raw_ds["train"].train_test_split(
            test_size=val_frac,
            stratify_by_column=label_col if label_col in raw_ds["train"].column_names else None,
            seed=seed,
        )
        raw_ds = DatasetDict(train=split["train"], validation=split["test"])
    else:
        raw_ds = DatasetDict(train=raw_ds["train"], validation=raw_ds["validation"])

    # 2) Take fractions (stratified when possible)
    def take_frac(ds, frac):
        if frac >= 1.0:  # keep full split
            return ds
        out = ds.train_test_split(
            test_size=1 - frac,
            stratify_by_column=label_col if label_col in ds.column_names else None,
            seed=seed,
        )
        return out["train"]  # the kept fraction

    small_train = take_frac(raw_ds["train"], train_frac)
    small_eval  = take_frac(raw_ds["validation"], val_frac)

    # 3) Tokenize (no padding here; we pad per-batch with the collator)
    def tok(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=max_length, padding=False)

    small_train_tok = small_train.map(tok, batched=True, remove_columns=[c for c in small_train.column_names if c not in (text_col, label_col)])
    small_eval_tok  = small_eval.map(tok,  batched=True, remove_columns=[c for c in small_eval.column_names  if c not in (text_col, label_col)])

    # 4) Keep only the columns Trainer needs
    keep_cols = ["input_ids", "attention_mask", label_col]
    small_train_tok = small_train_tok.remove_columns([c for c in small_train_tok.column_names if c not in keep_cols])
    small_eval_tok  = small_eval_tok.remove_columns([c for c in small_eval_tok.column_names  if c not in keep_cols])

    # 5) Data collator with dynamic padding (CUDA gets pad_to_multiple_of=8)
    import torch
    pad_to_mult = 8 if (pad_to_multiple_of_8_on_cuda and torch.cuda.is_available()) else None
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=pad_to_mult)

    return small_train_tok, small_eval_tok, data_collator, tokenizer


train_ds, eval_ds, data_collator, tokenizer = make_trainer_ready(
    raw_ds=dataset,
    model_name="cardiffnlp/twitter-roberta-base-sep2022",
    train_frac=0.2,    # take 20% of train
    val_frac=0.5,      # take 50% of validation
    seed=42,
    label_col="label",
    text_col="text",
    max_length=128,
)

# --- Training args: stop forking on macOS, fix pin_memory ---
trainer_fp16 = bool(device == "cuda" and use_fp16)
trainer_bf16 = bool(device == "cuda" and use_bf16)

training_args = TrainingArguments(
    output_dir="models/artifacts",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",

    eval_strategy="steps",
    logging_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_steps=500,

    load_best_model_at_end=True,
    metric_for_best_model="recall",
    greater_is_better=True,
    save_total_limit=2,

    # Precision
    fp16=trainer_fp16,
    bf16=trainer_bf16,

    # DataLoader knobs (avoid fork/tokenizers warning on macOS)
    dataloader_num_workers=0,                         # <- key for macOS/MPS
    dataloader_pin_memory=(device == "cuda"),         # False on MPS/CPU, True on CUDA
    group_by_length=True,
    report_to="none",
)

# --- Metrics (macro recall, etc.) ---
recall_metric = evaluate.load("recall")
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
        "recall": recall_metric.compute(predictions=preds, references=labels, average="macro")["recall"],
    }

callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= train_ds,
    eval_dataset= eval_ds,
    compute_metrics=compute_metrics,
    data_collator=data_collator,       # <- important
    tokenizer=tokenizer,
    callbacks=callbacks,
)

model.to(device)
trainer.train()
trainer.save_model("models/saved_model")
tokenizer.save_pretrained("models/saved_tokenizer")
try:
    trainer.create_model_card()
except Exception:
    pass
