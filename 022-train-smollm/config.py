"""
config.py
---------
Single source of truth for every constant used across the training pipeline.
Change a value here and it propagates everywhere.
"""

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

# ── Paths ────────────────────────────────────────────────────────────────────
DOCS_DIR   = "data/continue-pre-train"
OUTPUT_DIR = "google-ads-smollm"

# ── Document loading ─────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".md", ".mdc", ".sql", ".json"}

# ── Tokenisation ─────────────────────────────────────────────────────────────
BLOCK_SIZE = 512          # tokens per training example

# ── Training ─────────────────────────────────────────────────────────────────
NUM_TRAIN_EPOCHS            = 1
PER_DEVICE_BATCH_SIZE       = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE               = 2e-5
WEIGHT_DECAY                = 0.01
WARMUP_RATIO                = 0.05
LR_SCHEDULER_TYPE           = "cosine"
SAVE_STEPS                  = 100
LOGGING_STEPS               = 10
SAVE_TOTAL_LIMIT            = 2

# ── Evaluation ───────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 150

EVAL_PROMPTS = [
    "What is GAQL?",
    "Write a GAQL query to get campaign impressions for the last 30 days.",
    "What fields are available in the campaign resource?",
    "Explain the difference between a Search campaign and a Performance Max campaign.",
    "What is a bidding strategy in Google Ads?",
]
