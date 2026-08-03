# 008 — Residual Connections: Notes

## The Core Idea

```
output = x + sublayer(x)
```

Add the input back to the output.

## Two Problems Residuals Solve

### 1. Forward pass — information destruction

Without residuals, each block completely replaces the representation.
A later block can accidentally destroy what earlier blocks learned.

With residuals, every block **adds** a correction to the existing representation.
Earlier information is never fully erased.

### 2. Backward pass — vanishing gradients

Without shortcuts, gradients shrink through every layer:

```
0.9^50 ≈ 0.005
```

With shortcuts, the gradient through the residual path is:

```
∂L/∂x = ∂L/∂output × (1 + ∂sublayer/∂x)
```

The `+1` from the shortcut ensures the gradient always has a direct route back.

## The Intuition

> The sublayer only learns: "What should I change about this representation?"
> Not: "How do I rebuild everything from scratch?"

```
output ≈ x + small_correction
```

## In the Transformer Block

```
x = x + attention(layer_norm(x))   ← residual 1
x = x + ffn(layer_norm(x))         ← residual 2
```

Every block builds on the current representation, never replaces it.

## Pre-Norm vs Post-Norm

```
Post-Norm: LayerNorm(x + sublayer(x))     ← original Transformer
Pre-Norm:  x + sublayer(LayerNorm(x))     ← GPT-2, LLaMA
```

Pre-Norm is more stable at depth — LayerNorm stabilizes the input before each sublayer.

## History

ResNet (He et al., 2015) showed 152-layer image networks work when layers learn corrections.
Transformer (2017) adopted the same idea.

## Next

**Transformer Block** — combine multi-head attention + FFN + layer norm + residuals into one complete block.
