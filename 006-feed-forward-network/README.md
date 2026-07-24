# 006 — Feed-Forward Network

## Goal

By the end of this module, you should be able to answer:

* What is the feed-forward network in a Transformer?
* Why is it applied per-token independently?
* What activation function is used?
* What is the expansion factor and why?

---

# Theory

## 1. The Role of the FFN

After multi-head attention:

```
Each token now has a context-aware representation.
```

But attention is essentially **linear**.

The feed-forward network (FFN) introduces **non-linearity**.

It also gives the model more capacity to process each token.

---

## 2. Architecture

The FFN is applied **independently to each token**.

It is the same network, applied to each position:

```
For each token vector x:
   hidden = ReLU(x @ W1 + b1)
   output = hidden @ W2 + b2
```

This is a 2-layer MLP.

---

## 3. Dimensions

The hidden layer is expanded:

```
Input:  d_model  (e.g., 512)
Hidden: d_ff     (e.g., 2048)  ← 4× expansion
Output: d_model  (e.g., 512)
```

The expansion gives the network more capacity.

---

## 4. Activation Functions

Original Transformer: **ReLU**

```
ReLU(x) = max(0, x)
```

Modern models: **GELU**

```
GELU(x) ≈ x * Φ(x)   where Φ is the CDF of standard normal
```

GELU is smoother than ReLU.

Used in GPT-2, GPT-3, BERT.

---

## 5. Visual

```
Token vector: [0.21, 0.54, -0.13, ...]
                          ↓
              Linear W1 (d_model → d_ff)
                          ↓
              Activation (ReLU / GELU)
                          ↓
              Linear W2 (d_ff → d_model)
                          ↓
              Output: [0.33, -0.12, 0.88, ...]
```

Same operation applied to every token separately.

---

## 6. Why Per-Token?

Attention already mixed information across tokens.

FFN processes each token individually — it is a **position-wise** operation.

This is sometimes called "position-wise feed-forward."

---

# Coding Assignments

## Assignment 1 — ReLU Activation

Implement:

```python
def relu(x):
    pass
```

Test it on:

```python
[-2, -1, 0, 1, 2]
```

Expected:

```
[0, 0, 0, 1, 2]
```

---

## Assignment 2 — GELU Activation

Implement:

```python
def gelu(x):
    pass
```

Approximation:

```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

---

## Assignment 3 — Single FFN Forward Pass

Given one token vector `x` of shape `(d_model,)`:

```python
def ffn(x, W1, b1, W2, b2):
    pass
```

---

## Assignment 4 — Apply FFN to All Tokens

Input: `(seq_len, d_model)`
Output: `(seq_len, d_model)`

Apply FFN independently to each row.

---

# Success Criteria

* Know the FFN is a 2-layer MLP per token
* Know d_ff is typically 4× d_model
* Implement ReLU and GELU
* Know the input/output shape is unchanged
