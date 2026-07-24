# 010 — Transformer Stack: Notes

## Why Stack?

More layers → more abstract representations.

Early layers: local patterns (words, punctuation)
Late layers: global semantics (reasoning, facts)

## Full GPT Architecture

```
Token IDs
  ↓ Embedding
  ↓ + Positional Encoding
  ↓ Block 1 (MHA + FFN + LN + Residual)
  ↓ Block 2
  ↓ ...
  ↓ Block N
  ↓ Final LayerNorm
  ↓ LM Head (Linear → Softmax)
Output: token probabilities
```

## Real Model Configs

| Model | N | d_model | d_ff | h |
|-------|---|---------|------|---|
| GPT-2 Small | 12 | 768 | 3072 | 12 |
| GPT-3 | 96 | 12288 | 49152 | 96 |
| LLaMA 2 7B | 32 | 4096 | 11008 | 32 |

## Parameter Count Formula

```
per block ≈ 12 × d_model²
total ≈ N × 12 × d_model² + vocab_size × d_model
```

## Key Insight

The shape `(seq_len, d_model)` never changes through the stack.

All blocks see and output the same shape.

## Next

**Next Token Prediction** — add the LM head to turn context vectors into vocabulary probabilities.
