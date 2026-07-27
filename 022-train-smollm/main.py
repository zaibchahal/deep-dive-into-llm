"""
main.py
-------
Entry point for continued pre-training of SmolLM2-360M on Google Ads docs.

Pipeline:
  1. Load documents       (loader.py)
  2. Build HF Dataset     (dataset.py)
  3. Tokenize + chunk     (tokenizer_utils.py)
  4. Evaluate BEFORE      (evaluate.py)
  5. Train                (trainer.py)
  6. Save checkpoint      (transformers built-in)
  7. Evaluate AFTER       (evaluate.py)

Run:
    python main.py
    python main.py --epochs 3 --block-size 1024 --docs-dir docs
"""

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BLOCK_SIZE, DOCS_DIR, LEARNING_RATE, MODEL_NAME, OUTPUT_DIR
from config import (
    GRADIENT_ACCUMULATION_STEPS,
    NUM_TRAIN_EPOCHS,
    PER_DEVICE_BATCH_SIZE,
)
from dataset import build_dataset
from evaluate import run_eval
from loader import load_documents
from tokenizer_utils import tokenize_dataset
from trainer import build_trainer


def parse_args():
    p = argparse.ArgumentParser(description="Continued pre-training of SmolLM2")
    p.add_argument("--model", default=MODEL_NAME, help="HF model ID or local path")
    p.add_argument("--docs-dir", default=DOCS_DIR, help="Root directory for documents")
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Where to save the model")
    p.add_argument("--block-size", type=int, default=BLOCK_SIZE, help="Tokens per block")
    p.add_argument("--epochs", type=int, default=NUM_TRAIN_EPOCHS, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=PER_DEVICE_BATCH_SIZE, help="Per-device batch size")
    p.add_argument("--grad-accum", type=int, default=GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    p.add_argument("--skip-eval", action="store_true", help="Skip before/after eval")
    return p.parse_args()


def main():
    args = parse_args()

    docs_path = Path(args.docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(
            f"docs-dir '{docs_path}' not found.  "
            "Pass --docs-dir <path> or run from the 022-train-smollm directory."
        )

    # ── 1. Load tokenizer + model ────────────────────────────────────────
    print(f"\n[main] loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # SmolLM2 doesn't set a pad token by default; set it to EOS so the
    # DataCollator doesn't complain.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    print(f"[main] model parameters: {model.num_parameters():,}")

    # ── 2. Evaluate BEFORE training ──────────────────────────────────────
    if not args.skip_eval:
        print("\n[main] ── BEFORE TRAINING ──")
        run_eval(model, tokenizer, label="base model")

    # ── 3. Load documents ────────────────────────────────────────────────
    print("\n[main] loading documents ...")
    documents = load_documents(args.docs_dir)

    # ── 4. Build Dataset ─────────────────────────────────────────────────
    raw_dataset = build_dataset(documents)

    # ── 5. Tokenize + chunk ──────────────────────────────────────────────
    train_dataset = tokenize_dataset(raw_dataset, tokenizer, block_size=args.block_size)

    # ── 6. Train ─────────────────────────────────────────────────────────
    print("\n[main] building trainer ...")
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
    )

    print("\n[main] ── TRAINING START ──")
    trainer.train()

    # ── 7. Save ──────────────────────────────────────────────────────────
    print(f"\n[main] saving model to '{args.output_dir}' ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[main] saved ✓")

    # ── 8. Evaluate AFTER training ───────────────────────────────────────
    if not args.skip_eval:
        print("\n[main] ── AFTER TRAINING ──")
        run_eval(model, tokenizer, label="after training")

    print("\n[main] done.")


if __name__ == "__main__":
    main()
