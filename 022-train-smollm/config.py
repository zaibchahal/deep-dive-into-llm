"""
config.py
---------
Single source of truth for every constant used across the training pipeline.
Change a value here and it propagates everywhere automatically.
"""

# ── Model ─────────────────────────────────────────────────────────────────────
# The Hugging Face model ID to use as the starting point.
# SmolLM2-360M is a 360-million parameter causal language model from Hugging Face.
# It is small enough to train on a MacBook (Apple Silicon) and does not require
# a Hugging Face account or gated access.
# To switch to a larger model, just change this string — everything else adapts.
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"


# ── Paths ─────────────────────────────────────────────────────────────────────
# Root folder that loader.py walks recursively to find source documents.
# Change this to a smaller folder (e.g. "data/continue-pre-train") when you
# want a quick test run without loading the entire corpus.
DOCS_DIR = "data/continue-pre-train"

# Where the trained model and tokenizer are saved after training completes.
# This folder is created automatically by the Trainer.
# Pass it to evaluate.py with --model google-ads-smollm to test after training.
OUTPUT_DIR = "google-ads-smollm"


# ── Document loading ──────────────────────────────────────────────────────────
# File extensions that loader.py will read.
# Any file with an extension NOT in this set is silently ignored.
SUPPORTED_EXTENSIONS = {".md", ".mdc", ".sql", ".json"}


# ── Tokenisation ──────────────────────────────────────────────────────────────
# Number of tokens in each training example.
#
# How it works:
#   All documents are tokenized and concatenated into one long token stream.
#   That stream is then cut into non-overlapping windows of BLOCK_SIZE tokens.
#   Each window becomes one row in the training dataset.
#
# Trade-offs:
#   Smaller (e.g. 256) → more examples, faster per step, less context per example.
#   Larger  (e.g. 1024) → fewer examples, slower per step, model sees more context.
#   512 is a safe default for a 360M model on a MacBook.
BLOCK_SIZE = 512


# ── Training ──────────────────────────────────────────────────────────────────
# How many full passes over the training dataset to make.
# 1 epoch is standard for continued pre-training on a small corpus.
# More epochs risk overfitting (the model memorises your docs instead of generalising).
NUM_TRAIN_EPOCHS = 1

# How many training examples are processed in one forward+backward pass per device.
# Lower this (e.g. 2 or 1) if you run out of memory.
PER_DEVICE_BATCH_SIZE = 4

# Gradients are accumulated for this many steps before the optimiser takes one step.
# Effective batch size = PER_DEVICE_BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS = 4 × 4 = 16.
# This lets you simulate a larger batch without needing more RAM.
GRADIENT_ACCUMULATION_STEPS = 4

# How fast the model updates its weights.
# 2e-5 means 0.00002 — a small, careful step size suitable for continued pre-training.
# Too high → training becomes unstable (loss spikes).
# Too low  → training is very slow and may not converge.
LEARNING_RATE = 2e-5

# Penalty applied to large weights to prevent overfitting.
# 0.01 is a standard conservative value.
WEIGHT_DECAY = 0.01

# Fraction of total training steps used to linearly ramp the learning rate
# from 0 up to LEARNING_RATE before the cosine decay begins.
# 0.05 = the first 5% of steps are a warm-up phase.
# Helps avoid large destabilising updates at the very start of training.
WARMUP_RATIO = 0.05

# Shape of the learning rate curve after the warm-up phase.
# "cosine" → smoothly decays from LEARNING_RATE down to ~0 by the end of training.
# Other options: "linear", "constant", "polynomial".
LR_SCHEDULER_TYPE = "cosine"

# Save a checkpoint to OUTPUT_DIR every this many optimiser steps.
# A checkpoint lets you resume training if it is interrupted, or roll back
# to an earlier point if the loss starts going up.
SAVE_STEPS = 100

# Print the training loss to the console every this many optimiser steps.
# Lower values give more granular feedback; higher values produce less noise.
LOGGING_STEPS = 10

# Maximum number of checkpoints to keep on disk at any one time.
# Older checkpoints are deleted automatically when this limit is reached.
# 2 means you always have the current and one previous checkpoint.
SAVE_TOTAL_LIMIT = 2


# ── Evaluation ────────────────────────────────────────────────────────────────
# Maximum number of new tokens the model is allowed to generate per prompt
# during the before/after evaluation in evaluate.py.
MAX_NEW_TOKENS = 150

# Prompts used to compare the base model vs the fine-tuned model.
# These are intentionally Google Ads specific — if training worked, the
# fine-tuned model should answer these more accurately than the base model.
# Add or change prompts here to test whatever you care about most.
EVAL_PROMPTS = [
    "What is GAQL?",
    "Write a GAQL query to get campaign impressions for the last 30 days.",
    "What fields are available in the campaign resource?",
    "Explain the difference between a Search campaign and a Performance Max campaign.",
    "What is a bidding strategy in Google Ads?",
]
