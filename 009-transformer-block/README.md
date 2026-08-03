# 009 — Transformer Block

## References

* [How Transformers Work](https://www.nn-visual.com/transformers) — Interactive visualization of the full transformer pipeline. Explore the internals of a single block: multi-head attention → Add & Norm → feed-forward → Add & Norm, with real GPT-2 weights.

---

## Goal

By the end of this module, you should be able to answer:

* What is a Transformer block?
* What does one block improve about a token's representation?
* Why do we stack many Transformer blocks?
* What are its components and their order?
* What is the shape at each step?
* What is the difference between an encoder block and a GPT decoder-only block?

---

# Theory

## 1. Assembly

A Transformer block is not just a list of components — it is one complete **reasoning step** applied to every token.

Each block improves token representations in two ways:

1. **Attention** — let each token gather information from other tokens.
   *"What context from the sequence should I incorporate?"*

2. **FFN** — let each token analyze and transform its own features independently.
   *"Given what I now know, which features should I activate?"*

These two steps are wrapped in LayerNorm and residual connections so that each improvement is stable and cumulative.

One block moves every token's representation one step closer to being useful for predicting the next token.

---

## 2. GPT-Style Block (Decoder-only, Pre-Norm)

```text
Input embeddings
        │
        ▼
LayerNorm
(stabilize values before attention)
        │
        ▼
Multi-Head Self-Attention
("Which tokens should I look at?")
        │
        ▼
+ x  ◄── residual
("Keep old information + add improvement")
        │
        ▼
LayerNorm
(stabilize before FFN)
        │
        ▼
Feed-Forward Network
("Which features should activate?")
        │
        ▼
+ x  ◄── residual
("Keep old information + add improvement")
        │
        ▼
Output embeddings
(same shape — richer representation)
```

---

## 3. Detailed Step-by-Step

```python
# Input: x  shape (seq_len, d_model)

# --- Attention sublayer ---

x_norm   = LayerNorm(x)            # stabilize values

attn_out = MultiHeadAttention(x_norm)
# Each token looks at every other token and gathers context

x = x + attn_out
# Keep the original representation, add the learned correction

# --- FFN sublayer ---

x_norm  = LayerNorm(x)             # stabilize values again

ffn_out = FFN(x_norm)
# Each token independently analyzes its own features

x = x + ffn_out
# Keep the current representation, add another learned correction

# Output: x  shape (seq_len, d_model)   ← same shape as input
```

---

## 4. Shape Never Changes

The shape at every step is:

```
(seq_len, d_model)
```

This is not a coincidence — it is a design decision that makes stacking trivial:

```text
Embedding

(10, 768)
    │
    ▼
Transformer Block 1

(10, 768)
    │
    ▼
Transformer Block 2

(10, 768)
    │
    ▼
Transformer Block 3

(10, 768)
    │
    ▼
  ...

Block N

(10, 768)
```

Every block receives the same shape it outputs.
You can repeat the block dozens of times without changing anything else.
GPT-2 has 12 blocks. GPT-3 has 96.

---

## 5. Encoder vs GPT Decoder-Only Block

A common point of confusion:

> GPT is **not** generating text because it is a decoder.
>
> GPT is called decoder-only because it uses **causal (masked) attention** — each token can only see tokens that came before it.
>
> It does not contain an encoder at all.

|                        | Encoder (BERT) | GPT Decoder-only |
|------------------------|----------------|------------------|
| Sees future tokens     | Yes            | No               |
| Causal mask            | No             | Yes              |
| Has cross-attention    | No             | No               |
| Generates text         | No             | Yes              |
| Example                | BERT           | GPT, LLaMA       |

---

## 6. The Mental Model

Once the block structure is clear, the entire Transformer architecture reduces to four roles:

```text
Each token, at each block, does four things:

Attention:
"I learn from other tokens."

FFN:
"I analyze my own features."

Residual:
"I keep what I already knew."

LayerNorm:
"I keep the numbers stable."
```

Stack this process 12, 32, or 96 times and you have GPT-2, LLaMA-7B, or GPT-3.

---

## Advanced: Parameters in One Block

```
Multi-head attention:
  4 × (d_model × d_model) = 4d²

FFN:
  W1: d_model × d_ff = d × 4d = 4d²
  W2: d_ff × d_model = 4d × d = 4d²
  Total FFN: 8d²

Layer norms (2 per block):
  Each LayerNorm has gamma (d) + beta (d) = 2d parameters
  Two LayerNorms: 4d

Total per block ≈ 12d²  (the 4d term is negligible for large d)
```

GPT-2 (d=768, 12 blocks): ~85M parameters.

---

# Coding Assignments

## Assignment 0 — Trace One Token (No Code)

Before writing any code, trace what happens to the word `"cat"` in the sentence `"The cat sat"`.

```text
"cat" token embedding
        │
        ▼
Attention
→ cat looks at "The" and "sat"
→ gathers positional and relational context
→ produces correction to add to cat's embedding
        │
        ▼
Residual
→ cat's embedding = original + attention correction
        │
        ▼
FFN
→ cat analyzes its own updated features
→ activates feature detectors relevant to this token
→ produces another correction
        │
        ▼
Residual
→ cat's embedding = previous + FFN correction
        │
        ▼
Richer representation of "cat"
(still the same shape — ready for the next block)
```

Write this out in your own words before continuing.

---

## Assignment 1 — Assemble the Block

Using helper functions from previous modules:

```python
def transformer_block(x, params, use_causal_mask=True):
    # LayerNorm → Attention → Residual
    # LayerNorm → FFN → Residual
    pass
```

---

## Assignment 2 — Shape Verification

Verify input and output have the same shape for various sequence lengths.

---

## Assignment 3 — Print Intermediate Shapes

Print the shape at every step inside the block, and confirm it never changes.

---

# Success Criteria

* Understand that each block improves token representations — it does not replace them.
* Understand that attention changes a token using information from other tokens.
* Understand that FFN changes a token using only its own features.
* Understand that residuals keep previous information and add corrections.
* Understand that LayerNorm stabilizes values before each sublayer.
* Know the exact order of operations in a Transformer block.
* Implement the full block from components.
* Verify shape is preserved end-to-end.
