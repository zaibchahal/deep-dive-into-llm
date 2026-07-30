# 009 — Transformer Block

## References

* [How Transformers Work](https://www.nn-visual.com/transformers) — Interactive visualization of the full transformer pipeline. Explore the internals of a single block: multi-head attention → Add & Norm → feed-forward → Add & Norm, with real GPT-2 weights.

---

## Goal

By the end of this module, you should be able to answer:

* What is a Transformer block?
* What are its components and their order?
* What is the shape at each step?
* What is the difference between encoder and decoder blocks?

---

# Theory

## 1. Assembly

A Transformer block combines everything built so far:

1. Multi-Head Self-Attention
2. Residual Connection + Layer Norm
3. Feed-Forward Network
4. Residual Connection + Layer Norm

---

## 2. GPT-Style Block (Decoder-only, Pre-Norm)

```
x
↓
LayerNorm
↓
Multi-Head Self-Attention (with causal mask)
↓
+ x  (residual)
↓
LayerNorm
↓
Feed-Forward Network
↓
+ x  (residual)
↓
output
```

---

## 3. Detailed Step-by-Step

```
Input:  x  (shape: seq_len × d_model)

Step 1: Attention sublayer
  x_norm = LayerNorm(x)
  attn_out = MultiHeadAttention(x_norm)
  x = x + attn_out          ← residual 1

Step 2: FFN sublayer
  x_norm = LayerNorm(x)
  ffn_out = FFN(x_norm)
  x = x + ffn_out            ← residual 2

Output: x  (shape: seq_len × d_model)
```

---

## 4. Shape Never Changes

The input shape = output shape throughout:

```
(seq_len, d_model)
```

This is what allows stacking blocks.

---

## 5. Encoder vs Decoder Block

| | Encoder | Decoder |
|-|---------|---------|
| Attention mask | None | Causal (future masked) |
| Cross-attention | No | Yes (for seq2seq models) |
| Used in | BERT | GPT, LLaMA |

GPT and LLaMA are **decoder-only** — no encoder.

---

## 6. Parameters in One Block

```
Multi-head attention:
  4 × (d_model × d_model) = 4d²

FFN:
  d_model × d_ff + d_ff × d_model = 2 × d × 4d = 8d²

Layer norms:
  4 × d (gamma + beta, ×2)

Total per block ≈ 12d²
```

GPT-2 (d=768, 12 blocks): ~85M parameters.

---

# Coding Assignments

## Assignment 1 — Assemble the Block

Using your functions from previous modules:

```python
def transformer_block(x, ...):
    # LayerNorm + Attention + Residual
    # LayerNorm + FFN + Residual
    pass
```

---

## Assignment 2 — Shape Verification

Verify input and output have the same shape for various sequence lengths.

---

## Assignment 3 — Print Intermediate Shapes

Print the shape at every step inside the block.

---

# Success Criteria

* Know the exact order of operations in a Transformer block
* Implement the full block from components
* Verify shape is preserved end-to-end
