import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a QLoRA adapter for English to Russian translation.")
    parser.add_argument("--model_name_or_path", default="/kaggle/input/gemma-4-e2b-it/weights")
    parser.add_argument("--output_dir", default="/kaggle/working/enru-lora")

    parser.add_argument("--train_file", default=None, help="JSONL/JSON/CSV with source and target columns.")
    parser.add_argument("--dataset_name", default=None, help="Optional Hugging Face dataset name.")
    parser.add_argument("--dataset_config", default=None, help="Optional Hugging Face dataset config.")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--source_col", default="src")
    parser.add_argument("--target_col", default="tgt")
    parser.add_argument("--translation_langs", default="en,ru", help="For datasets with a translation dict.")

    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--max_seq_length", type=int, default=1536)
    parser.add_argument("--val_fraction", type=float, default=0.02)
    parser.add_argument("--group_size", type=int, default=1, help="Join this many adjacent pairs into paragraph-like examples.")

    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--logging_steps", type=int, default=25)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--resume", default=None, help="Path to checkpoint, or 'auto'.")
    return parser.parse_args()


def read_local_file(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return Dataset.from_list(rows)
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Dataset.from_list(data)
    if suffix == ".csv":
        return load_dataset("csv", data_files=path, split="train")
    raise ValueError(f"Unsupported train_file extension: {suffix}")


def load_training_dataset(args):
    if args.train_file:
        ds = read_local_file(args.train_file)
    elif args.dataset_name:
        ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    else:
        raise ValueError("Pass either --train_file or --dataset_name.")

    if args.max_samples and len(ds) > args.max_samples:
        ds = ds.shuffle(seed=42).select(range(args.max_samples))
    return ds


def extract_pair(example, args):
    if "translation" in example and isinstance(example["translation"], dict):
        src_lang, tgt_lang = [x.strip() for x in args.translation_langs.split(",", 1)]
        return example["translation"].get(src_lang), example["translation"].get(tgt_lang)

    if args.source_col in example and args.target_col in example:
        return example[args.source_col], example[args.target_col]

    for src_key, tgt_key in [("en", "ru"), ("source", "target"), ("english", "russian")]:
        if src_key in example and tgt_key in example:
            return example[src_key], example[tgt_key]

    raise KeyError(
        "Could not find translation fields. Use --source_col/--target_col, "
        "or a dataset with translation.en/translation.ru."
    )


def build_prompt(tokenizer, src, tgt=None):
    user_text = (
        "Переведи следующий английский абзац на русский язык.\n"
        "Ответ должен быть только на русском языке, без пояснений и комментариев.\n"
        "Сохрани имена, даты, числа, технические термины и порядок предложений.\n\n"
        f"Английский абзац:\n{src}\n\n"
        "Русский перевод:"
    )

    if tgt is None:
        messages = [{"role": "user", "content": user_text}]
    else:
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": tgt},
        ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=tgt is None)


def tokenize_dataset(ds, tokenizer, args):
    def convert(example):
        src, tgt = extract_pair(example, args)
        if not src or not tgt:
            return {"input_ids": [], "attention_mask": []}
        prompt_text = build_prompt(tokenizer, str(src).strip(), None)
        full_text = build_prompt(tokenizer, str(src).strip(), str(tgt).strip())
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        encoded = tokenizer(
            full_text,
            max_length=args.max_seq_length,
            truncation=True,
            padding=False,
        )
        labels = encoded["input_ids"].copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        encoded["labels"] = labels
        return encoded

    tokenized = ds.map(convert, remove_columns=ds.column_names)
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)
    return tokenized


def group_translation_pairs(ds, args):
    if args.group_size <= 1:
        return ds

    grouped = []
    buffer_src = []
    buffer_tgt = []

    for example in ds:
        try:
            src, tgt = extract_pair(example, args)
        except KeyError:
            continue
        if not src or not tgt:
            continue

        src = str(src).strip()
        tgt = str(tgt).strip()
        if not src or not tgt:
            continue

        buffer_src.append(src)
        buffer_tgt.append(tgt)

        if len(buffer_src) >= args.group_size:
            grouped.append({
                args.source_col: " ".join(buffer_src),
                args.target_col: " ".join(buffer_tgt),
            })
            buffer_src = []
            buffer_tgt = []

    if buffer_src:
        grouped.append({
            args.source_col: " ".join(buffer_src),
            args.target_col: " ".join(buffer_tgt),
        })

    return Dataset.from_list(grouped)


def latest_checkpoint(output_dir):
    path = Path(output_dir)
    checkpoints = sorted(path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None


class DataCollatorForCausalLMWithLabels:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]

        padded_labels = []
        for label in labels:
            if len(label) > max_len:
                label = label[:max_len]
            padded_labels.append(label + [-100] * (max_len - len(label)))

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    use_4bit = use_cuda and not args.no_4bit

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        device_map={"": 0} if use_cuda else None,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
    )

    if use_4bit:
        # Gemma 4 has very large non-quantized auxiliary tensors. PEFT's
        # prepare_model_for_kbit_training may cast them to fp32 and OOM on T4.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = load_training_dataset(args)
    ds = group_translation_pairs(ds, args)
    tokenized = tokenize_dataset(ds, tokenizer, args)
    split = tokenized.train_test_split(test_size=args.val_fraction, seed=42) if args.val_fraction else None
    train_ds = split["train"] if split else tokenized
    eval_ds = split["test"] if split else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        bf16=use_cuda,
        fp16=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_ds is not None else None,
        eval_strategy="steps" if eval_ds is not None else "no",
        save_total_limit=3,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForCausalLMWithLabels(tokenizer),
    )

    resume_from_checkpoint = args.resume
    if resume_from_checkpoint == "auto":
        resume_from_checkpoint = latest_checkpoint(args.output_dir)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
