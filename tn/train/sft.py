"""Qwen3-0.6B-Base 全参 SFT。

用法(经 launch.sh 后台跑,单卡):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m tn.train.sft \
      --data data/train_v0.jsonl --out runs/sft_v0

中断后加 --resume 从最近 checkpoint 续训。
"""

import argparse
import random

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from tn.constants import MODEL_ID
from tn.train.build_dataset import Collator, encode_example, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--global-batch", type=int, default=256)
    ap.add_argument("--per-device", type=int, default=64)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    recs = load_jsonl(args.data)
    random.Random(0).shuffle(recs)
    encoded = [e for r in recs if (e := encode_example(r, tok)) is not None]
    print(f"dataset: {len(recs)} recs -> {len(encoded)} encoded "
          f"(dropped {len(recs) - len(encoded)} overlong)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.per_device,
        gradient_accumulation_steps=max(1, args.global_batch // args.per_device),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=4,
        report_to=["tensorboard"],
        dataloader_num_workers=4,
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=encoded,
        data_collator=Collator(),
    )
    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(f"{args.out}/final")
    tok.save_pretrained(f"{args.out}/final")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
