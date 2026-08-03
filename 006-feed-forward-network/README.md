# 006 — Feed-Forward Network

## References

* [Neural Network Visualizer](https://projects.machcomputing.com/projects/neural-network) — Interactive backpropagation animation. Step through forward pass, backward pass, and weight updates in real time with presets for XOR, AND, OR, and more. Supports Sigmoid, ReLU, and Tanh activations with MSE, BCE, and Huber loss functions.

---

## Goal

By the end of this module, you should be able to answer:

* What is a neuron?
* What is the feed-forward network in a Transformer?
* Why is it applied per-token independently?
* What activation function is used, and why does it exist?
* Why is the hidden layer larger than the embedding?

---

# Theory

## 0. What Is a Neuron?

Before looking at the FFN, we need to answer one question.

> **A neuron is one set of weights that produces one output value.**

Given an input vector `x`, one neuron computes:

```
output = x[0]*w[0] + x[1]*w[1] + x[2]*w[2] + ... + b
```

That single number is the neuron's output.

The neuron asks: **"Does my pattern appear in this input?"**

If its weights happen to respond strongly to, say, "verb-like" features, then a large positive output means "yes, I detect something verb-like here."

The FFN stacks many of these neurons in parallel.

---

## 1. The Role of the FFN

After multi-head attention, each token has a **context-aware embedding** — a vector that now reflects not only the token itself, but also the tokens around it.

The FFN does **not** communicate with other tokens. Instead, it analyzes **one token at a time**.

Here is what the FFN does in sequence:

1. The first linear layer expands the embedding into many **neurons (feature detectors)**.
2. The activation function (ReLU/GELU) allows only the relevant feature detectors to become active.
3. The second linear layer combines the active features back into the original embedding size.

This gives the model much richer processing power than attention alone.

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

Here is what each step means:

```
Context-aware embedding
        │
        ▼
Linear W1
(expand: d_model → d_ff)
        │
        ▼
Many neurons (feature detectors)

Each neuron asks:
"Do I detect my feature in this token?"

        │
        ▼
Activation (ReLU / GELU)

Only relevant neurons stay active.
Neurons that detect nothing → output zeroed.

        │
        ▼
Linear W2
(compress: d_ff → d_model)

Combine all active features.

        │
        ▼
Updated embedding
```

---

## 3. Dimensions

```
Input : d_model  (e.g., 512)
Hidden: d_ff     (e.g., 2048)
Output: d_model  (e.g., 512)
```

Each hidden dimension corresponds to **one neuron**.

Expanding from `d_model` to `d_ff` gives the model many more neurons, allowing different neurons to specialize in detecting different features.

Think of these neurons as **specialists**:

> More specialists → richer feature detection.

One neuron might specialize in detecting "this token follows a preposition."
Another might specialize in "this token appears to be a number."
None of this is programmed — they learn it from data.

Many Transformers use `d_ff ≈ 4 × d_model`, although modern architectures sometimes use different expansion ratios.

---

## 4. Activation Functions

The activation function determines **which neurons are active**.

Without an activation function, the two linear layers would collapse into a single linear operation — and the model would have no way to detect complex, non-linear features.

**ReLU** turns negative values into zero, effectively telling a neuron:

> "You have nothing useful to contribute for this token."

```
Input:  [-3,  5, -2,  8]

          ↓  ReLU

Output: [ 0,  5,  0,  8]
```

Neurons with negative output are switched off. Only the active ones pass through.

**ReLU formula:**

```
ReLU(x) = max(0, x)
```

**GELU** is a smoother version — instead of a hard cutoff at zero, it gradually suppresses near-zero values.

```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

Used in GPT-2, GPT-3, BERT. Preferred in most modern models.

---

## 5. Visual

```
Context-aware embedding
        │
        ▼
Linear W1
(expand)
        │
        ▼
3072 neurons

Each neuron asks:

"Do I detect my feature?"

        │
        ▼
Activation

Only relevant neurons stay active

        │
        ▼
Linear W2

Combine all active features

        │
        ▼
Updated embedding
```

Same operation applied to every token separately.

---

## 6. Why Per-Token?

Attention already mixed information across tokens.

Now each token has everything it needs from its context.

The FFN examines each token individually — like a specialist examining **one patient at a time**.

> Attention gathers information from other patients (tokens).
>
> The FFN examines only the current patient.

This is sometimes called a **position-wise** feed-forward network.

---

# Coding Assignments

## Assignment 0 — One Neuron

Before building the FFN, implement a single neuron.

```python
def neuron(x, w, b):
    pass
```

Test it on:

```python
x = [2, 3, 4]
w = [1, -2, 0.5]
b = 1
```

Compute `x @ w + b` manually, then verify your function matches.

---

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

* Know that **one neuron = one set of weights producing one output.**
* Know that the FFN expands the embedding into many neurons (feature detectors).
* Know why activation functions are needed — without them, the two linear layers collapse into one.
* Know why the hidden layer is larger than the embedding — more neurons means more specialists.
* Know the FFN processes each token independently.
* Implement ReLU, GELU, and a simple FFN.
