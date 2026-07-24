# 010 — Transformer Stack

## Goal

By the end of this module, you should be able to answer:

* Why stack multiple Transformer blocks?
* How does depth relate to model capability?
* What is the full GPT architecture from input to output?
* What are N, d_model, d_ff, h in real models?

---

# Theory

## 1. One Block Is Not Enough

A single block can capture some patterns.

But language is complex.

Stacking N blocks allows:

* Lower layers: simple patterns (syntax, grammar)
* Middle layers: phrases and entities
* Upper layers: reasoning, long-range dependencies

---

## 2. The Stack

```
Token IDs
   ↓
Embedding Layer
   ↓
+ Positional Encoding
   ↓
Block 1
   ↓
Block 2
   ↓
...
   ↓
Block N
   ↓
Final LayerNorm
   ↓
LM Head (Linear → Softmax)
   ↓
Next token probabilities
```

---

## 3. Hyperparameters

| Model | N (layers) | d_model | d_ff | h (heads) | Params |
|-------|-----------|---------|------|-----------|--------|
| GPT-2 Small | 12 | 768 | 3072 | 12 | 117M |
| GPT-2 Large | 36 | 1280 | 5120 | 20 | 774M |
| GPT-3 | 96 | 12288 | 49152 | 96 | 175B |
| LLaMA 2 7B | 32 | 4096 | 11008 | 32 | 7B |

Depth (N) and width (d_model) both matter.

---

## 4. Parameter Count

Rough estimate:

```
params ≈ N × 12 × d_model²

GPT-2 Small:
  12 × 12 × 768²  = 85M   ✓
```

---

## 5. Information Flow

```
Block 1 sees raw token+position vectors.
Block 2 sees refined representations from Block 1.
Block 12 sees highly abstract features.
```

Each block refines the representation.

---

# Coding Assignments

## Assignment 1 — Stack N Blocks

```python
def transformer_stack(x, all_params, n_blocks):
    for i in range(n_blocks):
        x = transformer_block(x, all_params[i])
    return x
```

---

## Assignment 2 — Full Forward Pass

Combine:

1. Embedding lookup
2. Positional encoding
3. Transformer stack (N blocks)
4. Final LayerNorm

Input: token IDs `[3, 7, 2, 1]`
Output: context vectors `(seq_len, d_model)`

---

## Assignment 3 — Count Parameters

Count total parameters in the stack.

---

# Success Criteria

* Know why depth matters
* Build a stack of N blocks
* Implement the full transformer forward pass (excluding LM head)
* Know real model configurations
