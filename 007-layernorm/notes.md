# 007 — Layer Normalization: Notes

## Why Normalize?

Neural networks with large activations train poorly.

Layer norm keeps each token's vector in a stable range: mean≈0, std≈1.

## Formula

```
x_norm = (x - μ) / √(σ² + ε)
output = γ · x_norm + β
```

- μ = mean over features
- σ² = variance over features
- ε = numerical stability (1e-5)
- γ, β = learned scale and shift

## Batch Norm vs Layer Norm

Layer Norm works per-sample across features.
Batch Norm works per-feature across the batch.

Transformers need Layer Norm because batch sizes vary and sequences differ.

## Pre-Norm (modern) vs Post-Norm (original)

```
Post-Norm: LayerNorm(x + sublayer(x))
Pre-Norm:  x + sublayer(LayerNorm(x))
```

Pre-Norm is more stable → used in GPT-2, LLaMA.

## Initial Values

```
gamma = ones  (no scaling initially)
beta  = zeros (no shift initially)
```

Learned during training.

## Output

Same shape as input. Just re-scaled per token.

## Next

**Residual Connections** — add the input back to the output so gradients can flow freely.
