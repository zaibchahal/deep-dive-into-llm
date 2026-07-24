# 004 — Self-Attention

## Goal

By the end of this module, you should be able to answer:

* What is self-attention?
* What are Query, Key, and Value?
* How does the model decide which tokens to focus on?
* What is scaled dot-product attention?
* What is the attention matrix?

---

# Theory

## 1. The Problem: Tokens need context

Consider:

```
"The bank by the river was steep."
```

vs

```
"I went to the bank to deposit money."
```

The word `bank` means different things.

To understand `bank`, the model must look at other words in the sentence.

Self-attention lets each token **look at all other tokens** and decide how much to focus on each.

---

## 2. Query, Key, Value

Self-attention uses three learned matrices:

```
W_Q  (Query weights)
W_K  (Key weights)
W_V  (Value weights)
```

For each input token vector `x`:

```
Q = x @ W_Q   ← "What am I looking for?"
K = x @ W_K   ← "What do I contain?"
V = x @ W_V   ← "What do I give out?"
```

---

## 3. Attention Score

For each pair of tokens `(i, j)`:

```
score(i, j) = Q[i] · K[j]
```

This is the dot product.

High score = token `i` attends strongly to token `j`.

---

## 4. Scaled Dot-Product Attention

Formula:

```
Attention(Q, K, V) = softmax( Q @ K.T / sqrt(d_k) ) @ V
```

Step by step:

```
1.  scores = Q @ K.T          → shape: (seq_len, seq_len)

2.  scores = scores / sqrt(d_k)   ← prevents vanishing gradients

3.  weights = softmax(scores)  → rows sum to 1

4.  output = weights @ V       → weighted sum of values
```

---

## 5. Visual Example

Sentence: `"The cat sat"`

After Q, K, V projections:

```
Attention weights:

         The    cat    sat
The  [ 0.8,   0.1,   0.1  ]
cat  [ 0.2,   0.6,   0.2  ]
sat  [ 0.1,   0.7,   0.2  ]
```

Reading row "sat":

```
sat attends most to cat (0.7)
sat attends a little to The (0.1)
```

This makes sense — "sat" relates to "cat" (the subject).

---

## 6. Why Scale by sqrt(d_k)?

With large embedding dimensions:

```
d_k = 512
```

Dot products can become very large.

```
Q · K  →  large number
```

Softmax of large numbers becomes very sharp (near one-hot).

Gradient vanishes.

Dividing by `sqrt(d_k)` keeps values in a reasonable range.

---

## 7. Attention Is All You Need

This operation is **the core** of every Transformer.

```
GPT
BERT
LLaMA
Gemini
```

All use variants of this same attention formula.

---

# Coding Assignments

## Assignment 1 — Q, K, V Projections

Given:

```python
x = np.random.randn(4, 8)  # 4 tokens, dim=8
```

Create random weight matrices:

```python
W_Q = np.random.randn(8, 8)
W_K = np.random.randn(8, 8)
W_V = np.random.randn(8, 8)
```

Compute:

```python
Q = x @ W_Q
K = x @ W_K
V = x @ W_V
```

---

## Assignment 2 — Attention Scores

Compute:

```python
scores = Q @ K.T
```

Print the shape and values.

---

## Assignment 3 — Softmax

Implement softmax:

```python
def softmax(x):
    pass
```

Apply it row-wise to the scores.

---

## Assignment 4 — Scaled Dot-Product Attention

Implement:

```python
def attention(Q, K, V):
    pass
```

Return the output and attention weights.

---

## Assignment 5 — Causal Masking

In GPT, a token can only attend to **previous** tokens (not future ones).

Implement a causal mask:

```
mask:
[[0, -inf, -inf, -inf],
 [0,    0, -inf, -inf],
 [0,    0,    0, -inf],
 [0,    0,    0,    0]]
```

Add the mask before softmax.

---

# Success Criteria

* Understand Q, K, V conceptually
* Know the formula: `softmax(QK^T / sqrt(d_k)) V`
* Know why we scale
* Know what a causal mask is and why GPT uses it
