# 022 — Continued Pre-training on Google Ads Docs

Continued pre-training of **SmolLM2-360M** on a private Google Ads
documentation corpus.  The model already knows English; we teach it the
domain vocabulary, GAQL syntax, API fields, and diagnostic patterns by
running one more pass of next-token prediction on our own documents.

---

## Why SmolLM2 instead of Llama?

| | Llama 3.2 1B | SmolLM2-360M |
|---|---|---|
| Parameters | 1 B | 360 M |
| HF gated | Yes (requires approval) | No |
| Fits in 8 GB RAM | Tight | Comfortable |
| Good for learning CPT | Yes | Yes |

SmolLM2-360M is publicly available and trains comfortably on a MacBook
with Apple Silicon.

---

## Project structure

```
022-train-smollm/
│
├── config.py           ← All constants (model name, paths, hyperparams)
├── loader.py           ← Recursively reads docs into a list of strings
├── dataset.py          ← Wraps the list into a HF Dataset
├── tokenizer_utils.py  ← Tokenizes + chunks into fixed-length blocks
├── trainer.py          ← DataCollator + TrainingArguments + Trainer
├── evaluate.py         ← Before/after inference comparison
├── main.py             ← Full pipeline entry point
│
├── docs/               ← Source documents (not committed)
│   └── google-ads/
│       ├── api/
│       ├── diagnostics/
│       ├── db/
│       └── ...
│
└── google-ads-smollm/  ← Saved model (created after training)
```

---

## Setup

### 1. Install dependencies

```bash
pip install transformers torch accelerate datasets sentencepiece huggingface_hub
```

### 2. Activate your virtual environment

```bash
source ../venv/bin/activate
```

### 3. (Optional) Hugging Face login

SmolLM2 is public so no login is required.  If you switch to a gated
model (e.g. Llama), run:

```bash
huggingface-cli login
```

---

## Configuration

All tuneable values live in **`config.py`**.  Edit that file once and
every module picks up the change automatically.

| Constant | Default | Description |
|---|---|---|
| `MODEL_NAME` | `HuggingFaceTB/SmolLM2-360M` | Base model to continue pre-training |
| `DOCS_DIR` | `docs` | Root folder of source documents |
| `OUTPUT_DIR` | `google-ads-smollm` | Where the trained model is saved |
| `SUPPORTED_EXTENSIONS` | `.md .mdc .sql .json` | File types to load |
| `BLOCK_SIZE` | `512` | Tokens per training example |
| `NUM_TRAIN_EPOCHS` | `1` | Training epochs |
| `PER_DEVICE_BATCH_SIZE` | `4` | Batch size per device |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | Effective batch = batch × accum |
| `LEARNING_RATE` | `2e-5` | Peak learning rate |
| `WARMUP_RATIO` | `0.05` | Fraction of steps used for warm-up |
| `LR_SCHEDULER_TYPE` | `cosine` | Learning rate schedule |
| `SAVE_STEPS` | `100` | Save a checkpoint every N steps |
| `LOGGING_STEPS` | `10` | Log loss every N steps |
| `MAX_NEW_TOKENS` | `150` | Tokens generated during evaluation |

---

## Run the full pipeline

```bash
python main.py
```

This runs every step in order:

1. Load tokenizer + model
2. Evaluate the base model (before)
3. Load documents
4. Build HF Dataset
5. Tokenize + chunk into blocks
6. Train
7. Save model + tokenizer
8. Evaluate the fine-tuned model (after)

### Override defaults from the command line

```bash
# Use a different docs folder (e.g. a small test set)
python main.py --docs-dir data/continue-pre-train

# Train for 3 epochs with a smaller batch
python main.py --epochs 3 --batch-size 2

# Change block size and learning rate
python main.py --block-size 1024 --lr 1e-5

# Skip before/after eval (faster iteration)
python main.py --skip-eval

# Full set of flags
python main.py \
  --model   HuggingFaceTB/SmolLM2-360M \
  --docs-dir docs \
  --output-dir google-ads-smollm \
  --block-size 512 \
  --epochs 1 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-5
```

---

## Run individual steps

The steps below fall into two categories:

- **Data steps** — prepare and inspect data. Do not touch the model.
- **Training step** — the only step that updates model weights.

---

### Data Step 1 — Loader

Reads all supported files and prints a preview of the first document.

```bash
python loader.py                           # uses DOCS_DIR from config.py
python loader.py data/continue-pre-train   # or pass a custom path
```

Expected output:

```
[loader] loaded 170 documents from docs
First document preview (60574 chars): ...
```

---

### Data Step 2 — Dataset

Builds the HF Dataset and prints its schema and a sample row.

```bash
python dataset.py
```

Expected output:

```
[loader] loaded 170 documents from docs
[dataset] created dataset with 170 rows
Dataset({ features: ['text'], num_rows: 170 })
```

---

### Data Step 3 — Tokenizer utils

Tokenizes and chunks the full corpus into fixed-length blocks.
This only **prepares** the data — it does not touch the model.

```bash
python tokenizer_utils.py
```

Expected output:

```
[loader] loaded 170 documents from docs
[dataset] created dataset with 170 rows
Tokenizing: 100%|██████████| 170/170
Grouping into blocks: 100%|██████████| 170/170
[tokenizer_utils] 170 docs → 40561 blocks of 512 tokens each
```

> **Note:** you may see a warning about sequence length > 8192.
> This is harmless — the chunking step immediately slices the stream
> into 512-token windows.

---

### Step 1 — Evaluate the base model (before training)

See what the untrained model already knows. Run this before you train
so you have a baseline to compare against.

```bash
python evaluate.py
python evaluate.py --model HuggingFaceTB/SmolLM2-360M --label "base"
```

---

### Step 2 — Train the model ← this is the only step that updates weights

```bash
python main.py
```

You'll know it's training when you see the loss decreasing:

```
{'loss': 2.34, 'learning_rate': 1e-05, 'epoch': 0.1}
{'loss': 2.18, 'learning_rate': 8e-06, 'epoch': 0.2}
{'loss': 1.97, 'learning_rate': 5e-06, 'epoch': 0.5}
...
```

The model is saved to `google-ads-smollm/` when training completes.

---

### Step 3 — Evaluate the fine-tuned model (after training)

Compare the trained checkpoint against the base model baseline.

```bash
python evaluate.py --model google-ads-smollm --label "fine-tuned"
```

---

## What "continued pre-training" means

Standard pre-training teaches the model English on trillions of tokens.
We don't redo that.  We take the already-trained weights and run a few
more steps of the same objective — next-token prediction — but only on
our domain documents.  The model's general ability is preserved; it just
becomes better at predicting tokens that appear in Google Ads contexts
(GAQL queries, API field names, campaign types, etc.).

```
Base model           →  continued pre-training  →  Domain model
(knows English)         (sees our docs once)        (knows Google Ads)
```

---

## Difference from Ollama

| | Ollama | Transformers (this project) |
|---|---|---|
| Format | GGUF (quantized) | Original PyTorch weights |
| Purpose | Inference only | Training + Inference |
| Can retrain? | No | Yes |
| RAM usage | Lower | Higher |

After training here, you can convert the saved model to GGUF with
`llama.cpp` and load it in Ollama for fast local inference.
