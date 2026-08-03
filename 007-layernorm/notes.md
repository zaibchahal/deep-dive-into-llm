# 007 — Layer Normalization: Notes

## The Problem

After Attention and FFN, token vectors can have wildly different scales.

The next layer can't learn well when inputs vary from `[0.5, -0.2, 1.1]` to `[500, -200, 900]`.
Large values → unstable gradients → harder training.

## What LayerNorm Does

Normalizes **each token vector independently**:

1. Subtract mean → centered around zero
2. Divide by std → consistent spread

```text
[8, 10, 12]
  → subtract mean (10) → [-2, 0, 2]
  → divide by std (1.63) → [-1.22, 0, 1.22]
```

The relative pattern is preserved. The scale is removed.

## Formula

```
μ      = mean(x)
σ²     = variance(x)
x_norm = (x - μ) / √(σ² + ε)
output = γ · x_norm + β
```

- `ε` = small constant for numerical stability (e.g. 1e-5)
- `γ, β` = learned scale and shift (initialized to 1 and 0)

## Why Gamma and Beta?

After normalization, the model has no control over scale.

Gamma and beta give the model learnable flexibility to rescale and reshift the normalized output.

At initialization `γ=1, β=0` → output equals `x_norm`.
During training the model learns the optimal scale and shift for each dimension.

> Normalization removes uncontrolled variation.
> Gamma and beta restore controlled flexibility.

## LayerNorm vs BatchNorm

| | BatchNorm | LayerNorm |
|---|-----------|-----------|
| Normalizes over | batch dimension | feature dimension |
| Depends on batch size | Yes | No |
| Works with 1 example | No | Yes |
| Used in | CNNs | Transformers |

LayerNorm normalizes within a single token. BatchNorm normalizes across examples.
Transformers use LayerNorm because sequences vary in length and inference often uses a single example.

## Pre-Norm vs Post-Norm

```
Post-Norm (2017): LayerNorm(x + sublayer(x))
Pre-Norm  (GPT-2, LLaMA): x + sublayer(LayerNorm(x))
```

Pre-Norm applies normalization **before** the sublayer.
This stabilizes very deep networks — preferred in modern Transformers.

## Shape

Input and output are always the same shape.
LayerNorm never changes the embedding size.

## Next

**Residual Connections** — add the input back to the output so gradients can flow freely through deep networks.
