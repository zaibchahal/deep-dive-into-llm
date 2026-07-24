# 009 — Transformer Block: Notes

## The Complete Block (Pre-Norm / GPT Style)

```
x → LayerNorm → MultiHeadAttention → + x
  → LayerNorm → FFN               → + x
```

## Order of Operations

1. LayerNorm(x)
2. MultiHeadAttention → attn_out
3. x = x + attn_out        (residual 1)
4. LayerNorm(x)
5. FFN → ffn_out
6. x = x + ffn_out         (residual 2)

## Shape Throughout

```
(seq_len, d_model)   input
(seq_len, d_model)   after LN 1
(seq_len, d_model)   after attention
(seq_len, d_model)   after residual 1
(seq_len, d_model)   after LN 2
(seq_len, d_model)   after FFN
(seq_len, d_model)   after residual 2 = output
```

Shape never changes — this allows stacking N blocks.

## Parameters per Block

```
≈ 12 × d_model²
```

For d_model=768 (GPT-2): ~7M params per block × 12 blocks = ~85M.

## Causal Mask

GPT blocks use a causal mask in attention so each token only sees past tokens.

## Next

**Transformer Stack** — stack N of these blocks to build the full model depth.
