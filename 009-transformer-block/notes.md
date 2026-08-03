# 009 — Transformer Block: Notes

## The Mental Model

One block = one reasoning step applied to every token.

```
Attention:  "I learn from other tokens."
FFN:        "I analyze my own features."
Residual:   "I keep what I already knew."
LayerNorm:  "I keep the numbers stable."
```

Stack this 12, 32, or 96 times → GPT-2, LLaMA-7B, GPT-3.

## The Complete Block (Pre-Norm / GPT Style)

```
x → LayerNorm → MultiHeadAttention → +x
  → LayerNorm → FFN               → +x
```

## Order of Operations

```python
# Attention sublayer
x_norm   = LayerNorm(x)
attn_out = MultiHeadAttention(x_norm)   # gather context from other tokens
x        = x + attn_out                 # keep original, add correction

# FFN sublayer
x_norm  = LayerNorm(x)
ffn_out = FFN(x_norm)                   # analyze own features
x       = x + ffn_out                   # keep current, add correction
```

## Shape Throughout

```
(seq_len, d_model)   input
(seq_len, d_model)   after LN 1
(seq_len, d_model)   after attention
(seq_len, d_model)   after residual 1
(seq_len, d_model)   after LN 2
(seq_len, d_model)   after FFN
(seq_len, d_model)   after residual 2  ← output
```

Shape never changes. This is what allows stacking N blocks.

## Encoder vs GPT Decoder-only

GPT is called decoder-only because it uses causal (masked) attention.
It is NOT because it decodes in the seq2seq sense — it has no encoder.

|                     | Encoder (BERT) | GPT Decoder-only |
|---------------------|----------------|------------------|
| Sees future tokens  | Yes            | No               |
| Causal mask         | No             | Yes              |
| Generates text      | No             | Yes              |

## Parameters per Block (approx)

```
≈ 12 × d_model²
```

Breakdown:
- Attention (4 weight matrices): 4d²
- FFN (W1 + W2): 8d²
- LayerNorm ×2: 4d (negligible)

GPT-2 (d=768, 12 blocks): ~85M parameters.

## Next

**Transformer Stack** — stack N of these blocks to build the full model depth.
