# 003 — Positional Encoding

## Goal

By the end of this module, you should be able to answer:

* Why do Transformers need positional encoding?
* What is sinusoidal positional encoding?
* How does the model know token order?
* What is the difference between absolute and learned positional encoding?

---

# Theory

## 1. The Problem: Transformers have no memory of order

From 002, we know:

```
"The dog bites the man"
```

Tokenizer:

```
The   → 15
dog   → 72
bites → 301
the   → 15
man   → 88
```

Embedding layer:

```
[
  [0.12, 0.44, ...],   ← "The"
  [0.21, 0.54, ...],   ← "dog"
  [0.33, -0.21, ...],  ← "bites"
  [0.12, 0.44, ...],   ← "the"
  [0.77, 0.11, ...],   ← "man"
]
```

Now consider:

```
"The man bites the dog"
```

The embeddings for the **same words** are identical.

But the **meaning is completely different**.

A Transformer processes all tokens in **parallel** — it does not read left to right.

Without position info, it cannot tell:

```
dog → position 2
```

vs

```
dog → position 5
```

---

## 2. Solution: Add Position to Embedding

We create a **positional encoding** vector for each position.

Then we **add** it to the word embedding:

```
final_input = word_embedding + positional_encoding
```

Example:

```
"dog" at position 2:

word_embedding:       [0.21, 0.54, -0.13, 0.87]
positional_encoding:  [0.00, 1.00,  0.00, 1.00]
                    +
                    =
final_input:          [0.21, 1.54, -0.13, 1.87]
```

Now the model sees different vectors for:

```
"dog" at position 2
```

vs

```
"dog" at position 5
```

---

## 3. Sinusoidal Positional Encoding (Original Transformer)

From the original "Attention Is All You Need" paper (2017):

Formula:

```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
```

Where:

```
pos    = position in sequence (0, 1, 2, ...)
i      = dimension index
d_model = embedding dimension
```

Why sinusoidal?

* No parameters to learn
* Can extrapolate to sequences longer than training
* Each position gets a unique pattern

---

## 4. Visual: Positional Encoding Matrix

For a sequence of length 4 and embedding dim 8:

```
Position 0:  [sin(0), cos(0), sin(0), cos(0), ...]
Position 1:  [sin(1), cos(1), sin(0.1), cos(0.1), ...]
Position 2:  [sin(2), cos(2), sin(0.2), cos(0.2), ...]
Position 3:  [sin(3), cos(3), sin(0.3), cos(0.3), ...]
```

Each row is a unique fingerprint for that position.

---

## 5. Learned Positional Encoding

Modern models (GPT-2, GPT-3) use **learned** positional embeddings.

Instead of a formula:

```
position_embedding = nn.Embedding(max_seq_len, d_model)
```

The model learns the best positional representation during training.

---

## 6. Full Input to Transformer

```
Token IDs
   ↓
Embedding Lookup
   ↓
Word Embeddings
   +
Positional Encoding
   ↓
Final Input Vectors
   ↓
Transformer Blocks
```

---

# Coding Assignments

## Assignment 1 — Manual Positional Encoding

For sequence length 4, embedding dim 4:

Create a matrix of shape `(4, 4)` manually:

```
position 0: [0, 1, 0, 1]
position 1: [sin(1), cos(1), sin(0.1), cos(0.1)]
position 2: [sin(2), cos(2), sin(0.2), cos(0.2)]
position 3: [sin(3), cos(3), sin(0.3), cos(0.3)]
```

Print the matrix.

---

## Assignment 2 — Sinusoidal PE Function

Implement:

```python
def positional_encoding(seq_len, d_model):
    pass
```

Use the formula:

```
PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
```

Output shape: `(seq_len, d_model)`

---

## Assignment 3 — Add to Embeddings

Given:

```python
word_embeddings = np.random.randn(4, 8)  # 4 tokens, dim 8
```

Add positional encoding:

```python
final_input = word_embeddings + positional_encoding(4, 8)
```

Print the result.

---

## Assignment 4 — Visualize Positions Differ

Show that two sequences with the same tokens but different positions produce different final inputs.

---

# Success Criteria

By the end of this module, you should understand:

* Why order matters and how Transformers lose it
* How PE is added to word embeddings
* The sinusoidal formula and why it works
* The difference between fixed (sinusoidal) and learned PE
