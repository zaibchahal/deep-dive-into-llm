# 004 — Self-Attention: Notes

## The Core Idea

Every token asks: *"Which other tokens should I pay attention to?"*

The answer is computed by comparing queries to keys.

## Q, K, V Intuition

- **Query**: what this token is looking for
- **Key**: what this token can offer to others
- **Value**: the actual content passed forward

Think of it like a search engine:
- Query = search term
- Key = document title
- Value = document content

## The Formula

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
```

Step by step:
1. `QKᵀ` → dot product → similarity scores
2. `/ √d_k` → scale to prevent vanishing gradients
3. `softmax` → convert to probabilities (sum to 1)
4. `× V` → weighted combination of values

## Causal Masking (GPT-style)

GPT only looks at past tokens — not future ones.

Upper triangle of attention matrix is set to `-inf`.

After softmax, `-inf` → 0 (no attention to future).

```
Token 0: sees only token 0
Token 1: sees tokens 0, 1
Token 2: sees tokens 0, 1, 2
...
```

## Why Scale?

Without scaling, large `d_k` causes large dot products.

Large inputs → softmax saturates → gradient vanishes.

`√d_k` keeps the scale reasonable.

## Output Shape

Input: `(seq_len, d_model)`
Output: `(seq_len, d_model)`

Same shape — self-attention is a feature transformation.

## Next

**Multi-Head Attention** — run multiple attention heads in parallel for richer representations.
