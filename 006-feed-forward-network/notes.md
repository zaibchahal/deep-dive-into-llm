# 006 — Feed-Forward Network: Notes

## What It Is

A simple 2-layer MLP applied independently to each token.

```
FFN(x) = activation(x @ W1 + b1) @ W2 + b2
```

## Why It Exists

- Attention is linear (weighted sums)
- FFN adds non-linearity
- Gives the model more capacity per token
- Attention: mix across tokens. FFN: process each token alone.

## Dimensions

```
d_model = 512
d_ff    = 2048  (4× expansion)
```

Expansion gives capacity. Compression forces useful representation.

## Activations

| Activation | Formula | Used In |
|------------|---------|---------|
| ReLU | max(0, x) | Original Transformer |
| GELU | ≈ x·Φ(x) | GPT-2, BERT |
| SwiGLU | x·σ(Wx)·Vx | LLaMA |

## Shape

```
Input:  (seq_len, d_model)
→ W1:   (seq_len, d_ff)
→ act:  (seq_len, d_ff)
→ W2:   (seq_len, d_model)
Output: (seq_len, d_model)
```

Input and output shapes match.

## Key Insight

FFN is position-wise: the **same** FFN is applied to every token.

No information flows between tokens in the FFN — that already happened in attention.

## Next

**Layer Normalization** — stabilizes training by normalizing each token's vector.
