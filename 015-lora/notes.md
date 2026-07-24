# 015 — LoRA: Notes

## Core Idea

Instead of fine-tuning all weights W, inject a small trainable perturbation:

```
output = x @ W + scale × x @ A.T @ B.T
```

W is frozen. Only A and B are trained.

## Why It Works

Fine-tuning changes are empirically low-rank.

A low-rank matrix captures most of the useful signal.

## Math

```
ΔW = B × A    (rank r, much smaller than W)

params(ΔW) = r×d_in + d_out×r = 2rd
params(W)  = d_in × d_out = d²

For d=4096, r=16: 2×16×4096 = 131K vs 16M (128× savings)
```

## Initialization

```
A ~ N(0, small)
B = 0
```

ΔW = B@A = 0 at start → model starts as pretrained model.

## Hyperparameters

- **r**: rank. Common values: 4, 8, 16, 64.
- **alpha**: scaling factor. Often set equal to r.
- **scale** = alpha / r

## Which Layers

Applied to: W_Q, W_V (at minimum), often W_K, W_O too.

NOT applied to: LayerNorm, embeddings, biases.

## QLoRA

LoRA + 4-bit quantized base model.

Enables fine-tuning 70B models on a single consumer GPU.

## Next

**RAG** — retrieval-augmented generation: give the model access to external knowledge without fine-tuning.
