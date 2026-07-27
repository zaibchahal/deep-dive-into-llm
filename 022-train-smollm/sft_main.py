"""
sft_main.py
-----------
Entry point for SFT (Supervised Fine-Tuning) of SmolLM2 on
GAQL instruction-response examples.

This runs AFTER continued pre-training.  The recommended order is:

  1. python main.py          ← CPT: teach the model Google Ads vocabulary
  2. python sft_main.py      ← SFT: teach the model to follow instructions

Pipeline:
  1. Load tokenizer + model (from CPT checkpoint or base model)
  2. Load + format SFT examples (chat template applied)
  3. Train with SFTTrainer   (loss only on assistant response tokens)
  4. Save the SFT model
  5. Compare base vs SFT with eval prompts

Run:
    python sft_main.py
    python sft_main.py --model google-ads-smollm   # start from CPT checkpoint
    python sft_main.py --skip-eval
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    MODEL_NAME,
    SFT_BLOCK_SIZE,
    SFT_DATA_PATH,
    SFT_NUM_TRAIN_EPOCHS,
    SFT_OUTPUT_DIR,
    SFT_PER_DEVICE_BATCH_SIZE,
)
from evaluate import run_eval
from sft_dataset import build_sft_dataset, load_sft_examples
from sft_trainer import build_sft_trainer


def parse_args():
    p = argparse.ArgumentParser(description="SFT fine-tuning of SmolLM2 on GAQL examples")
    p.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Base model or CPT checkpoint to start from (default: base SmolLM2)",
    )
    p.add_argument("--data-path", default=SFT_DATA_PATH, help="Path to JSONL SFT examples")
    p.add_argument("--output-dir", default=SFT_OUTPUT_DIR, help="Where to save the SFT model")
    p.add_argument("--epochs", type=int, default=SFT_NUM_TRAIN_EPOCHS, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=SFT_PER_DEVICE_BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--max-seq-length", type=int, default=SFT_BLOCK_SIZE)
    p.add_argument("--skip-eval", action="store_true", help="Skip before/after eval")
    return p.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"SFT data not found: {data_path}\n"
            "Add examples to data/sft/sft_examples.jsonl and try again."
        )

    # ── 1. Load tokenizer + model ────────────────────────────────────────
    print(f"\n[sft_main] loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
    )
    print(f"[sft_main] model parameters: {model.num_parameters():,}")
    print(f"[sft_main] model dtype: {dtype}")

    # ── 2. Evaluate BEFORE ───────────────────────────────────────────────
    if not args.skip_eval:
        print("\n[sft_main] ── BEFORE SFT ──")
        run_eval(model, tokenizer, label="before SFT")

    # ── 3. Load + format SFT examples ───────────────────────────────────
    print(f"\n[sft_main] loading SFT examples from {args.data_path} ...")
    examples = load_sft_examples(args.data_path)
    train_dataset = build_sft_dataset(examples, tokenizer)

    # ── 4. Train ─────────────────────────────────────────────────────────
    print("\n[sft_main] building SFT trainer ...")
    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
    )

    print("\n[sft_main] ── SFT TRAINING START ──")
    trainer.train()

    # ── 5. Save ──────────────────────────────────────────────────────────
    print(f"\n[sft_main] saving model to '{args.output_dir}' ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[sft_main] saved ✓")

    # ── 6. Evaluate AFTER ────────────────────────────────────────────────
    if not args.skip_eval:
        print("\n[sft_main] ── AFTER SFT ──")
        run_eval(model, tokenizer, label="after SFT")

    print("\n[sft_main] done.")


if __name__ == "__main__":
    main()
