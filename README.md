# AI/ML Course — From Tokens to RLHF

A ground-up course building understanding of how Large Language Models work, from the smallest pieces to modern training techniques.

## Structure

Each module contains:
- `README.md` — theory + coding assignments
- `main.py` — implementations
- `test.py` — tests
- `notes.md` — key takeaways

## Modules

| # | Topic | Key Concept |
|---|-------|-------------|
| 001 | Tokenization | Text → Token IDs |
| 002 | Embeddings | IDs → Dense Vectors |
| 003 | Positional Encoding | Inject order into embeddings |
| 004 | Self-Attention | Q, K, V — scaled dot-product |
| 005 | Multi-Head Attention | Parallel attention heads |
| 006 | Feed-Forward Network | Per-token MLP with activation |
| 007 | Layer Normalization | Stabilize training |
| 008 | Residual Connections | Skip connections for gradient flow |
| 009 | Transformer Block | Assemble the full block |
| 010 | Transformer Stack | Stack N blocks |
| 011 | Next Token Prediction | LM head, cross-entropy loss |
| 012 | Decoding | Greedy, temperature, top-k, top-p |
| 013 | Train Tiny GPT | Full training loop from scratch |
| 014 | Inference | KV cache, batch inference |
| 015 | LoRA | Parameter-efficient fine-tuning |
| 016 | RAG | Retrieval-augmented generation |
| 017 | Tool Calling | LLM + external functions |
| 018 | Agent | ReAct loop: observe-think-act |
| 019 | SFT | Instruction tuning with loss masking |
| 020 | RLHF Basics | Reward model, DPO, preference alignment |
| 021 | Hallucination Mitigation | Self-consistency, entropy, faithfulness, abstention |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirments.txt
```

## Running Tests

```bash
cd 003-positional-encoding
python test.py
```

## Prerequisites

- Python 3.8+
- NumPy
- tiktoken (for module 001)
