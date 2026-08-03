# 006 — Feed-Forward Network: Notes

## What Is a Neuron?

> A neuron is one set of weights that produces one output value.

```
output = x[0]*w[0] + x[1]*w[1] + ... + b
```

The neuron asks: "Does my pattern appear in this input?"

---

## What It Is

A simple 2-layer MLP applied independently to each token.

```
FFN(x) = activation(x @ W1 + b1) @ W2 + b2
```

## Why It Exists

- Attention mixes information across tokens — but it is essentially linear (weighted sums).
- FFN adds non-linearity via the activation function.
- FFN gives each token a richer per-token processing step.
- Attention: mix across tokens. FFN: process each token alone.

## Dimensions

```
d_model = 512
d_ff    = 2048  (4× expansion — common, but not universal)
```

Each hidden dimension = one neuron = one specialist.

Expanding from `d_model` to `d_ff` gives the model many more specialists, each able to detect a different feature.

Compression back to `d_model` forces the model to summarize what it found.

## Activations

| Activation | Formula | Used In |
|------------|---------|---------|
| ReLU  | max(0, x) | Original Transformer |
| GELU  | ≈ x·Φ(x) | GPT-2, BERT |
| SwiGLU | x·σ(Wx)·Vx | LLaMA |

### Intuition

ReLU hard-zeroes any neuron whose output is negative.
GELU softly suppresses near-zero values instead.

Without activation, two linear layers collapse into one — no non-linearity, no ability to detect complex features.

```
Input:  [-3,  5, -2,  8]
          ↓  ReLU
Output: [ 0,  5,  0,  8]
```

Only active neurons pass their value through.

## Shape

```
Input:  (seq_len, d_model)
→ W1:   (seq_len, d_ff)
→ act:  (seq_len, d_ff)
→ W2:   (seq_len, d_model)
Output: (seq_len, d_model)
```

Input and output shapes match. The FFN never changes the embedding size.

## Key Insight

FFN is position-wise: the **same** FFN is applied to every token.

No information flows between tokens in the FFN — that already happened in attention.

> Attention gathers information from other tokens.
> The FFN examines only the current token.

## Next

**Layer Normalization** — stabilizes training by normalizing each token's vector.
