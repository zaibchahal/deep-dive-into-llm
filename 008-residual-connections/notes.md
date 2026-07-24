# 008 — Residual Connections: Notes

## The Core Idea

```
output = x + sublayer(x)
```

Add the input back to the output.

This gives gradients a shortcut during backpropagation.

## Why It Works

Backprop gradient through residual:

```
∂L/∂x = ∂L/∂output × (1 + ∂sublayer/∂x)
```

The `+1` means gradient never vanishes, even if sublayer gradients are tiny.

## History

ResNets (2015) used residuals to train 152-layer image networks.

Transformers adopted them for the same reason.

## In the Transformer Block

```
x = x + self_attention(layer_norm(x))   ← residual 1
x = x + ffn(layer_norm(x))              ← residual 2
```

## Pre-Norm vs Post-Norm

```
Post-Norm: LayerNorm(x + sublayer(x))     ← original Transformer
Pre-Norm:  x + sublayer(LayerNorm(x))     ← GPT-2, LLaMA
```

Pre-Norm trains more stably at depth.

## Intuition

The sublayer only needs to learn small corrections.

```
output ≈ x + small_improvement
```

Easier optimization than learning the full transformation from scratch.

## Next

**Transformer Block** — combine multi-head attention + FFN + layer norm + residuals into one complete block.
