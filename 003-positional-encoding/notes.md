# 003 — Positional Encoding: Notes

## The Core Problem

Transformers process all tokens in parallel.

No recurrence. No left-to-right reading.

Without position info, `"dog bites man"` == `"man bites dog"`.

## Solution

Add a position vector to each word embedding:

```
final = word_embedding + positional_encoding
```

## Sinusoidal Formula

```
PE[pos, 2i]   = sin(pos / 10000^(2i/d))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
```

- Even dimensions: sine
- Odd dimensions: cosine
- Each position gets a unique vector

## Why 10000?

Large constant spreads frequencies across dimensions.

Low dimensions → high frequency (changes fast with position)
High dimensions → low frequency (changes slowly)

This gives the model information at multiple scales.

## Fixed vs Learned PE

| Fixed (Sinusoidal) | Learned |
|--------------------|---------|
| No parameters      | Trainable params |
| Extrapolates to longer seqs | Limited to training length |
| Original Transformer | GPT-2, GPT-3 |

## Key Equation

```
input_to_transformer = word_embedding + positional_encoding
```

Shape: `(seq_len, d_model)`

## Next

**Self-Attention** — now that the model has position-aware vectors, it learns which tokens to attend to.
