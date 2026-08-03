# 007 — Layer Normalization

## Goal

By the end of this module, you should be able to answer:

* What problem does LayerNorm solve?
* Why does a Transformer normalize each token independently?
* How does LayerNorm make embeddings stable?
* Why are gamma and beta needed?
* What is the difference between BatchNorm and LayerNorm?
* What are Pre-Norm and Post-Norm architectures?

---

# Theory

## 1. The Problem: Unstable Activations

After Attention and FFN, token representations can have very different scales.

Example:

```text
Token A:  [0.5, -0.2,  1.1,  0.8]

Token B:  [500, -200,  900,  300]
```

The problem is not that the information is wrong.

The problem is:

> The next layer receives numbers with inconsistent scales.

Large values can cause:

* unstable gradients
* harder optimization
* slower training

---

## 2. What LayerNorm Does (Intuition First)

LayerNorm makes each token vector:

1. **Centered around zero** — subtract the mean
2. **Consistent spread** — divide by the standard deviation

Example:

Before:

```text
[8, 10, 12]
```

Subtract the mean (mean = 10):

```text
[-2, 0, 2]
```

Now centered. But the spread still varies between tokens.

Divide by spread (std ≈ 1.63):

```text
[-1.22, 0, 1.22]
```

Now centered **and** same scale — regardless of the original magnitudes.

The relative pattern is preserved. The scale is not.

---

## 3. LayerNorm Formula

For one token vector `x = [x1, x2, ... xd]`:

**Mean** — center the values:

```
μ = (x1 + x2 + ... + xd) / d
```

**Variance** — measure the spread:

```
σ² = average of (xi - μ)²
```

**Normalize:**

```python
x_norm = (x - μ) / sqrt(σ² + ε)
```

Result:

```
mean ≈ 0
variance ≈ 1
```

`ε` is a tiny constant (e.g. `1e-5`) added to prevent division by zero when variance is near zero.

---

## 4. Why Gamma and Beta Exist

After normalization, every token vector looks like this:

```text
[-1.2, 0, 1.2]
```

The model has **no control over the scale anymore**.

But maybe the next layer works better with a larger or shifted range:

```text
[-3.6, 0, 3.6]    ← scaled by 3
```

or:

```text
[-0.2, 1.0, 2.2]  ← shifted by 1
```

So LayerNorm adds two **learnable** parameters per dimension:

```python
output = gamma * x_norm + beta
```

Where:

```
gamma = learned scale   (initialized to 1)
beta  = learned shift   (initialized to 0)
```

At initialization:

```python
output = 1 * x_norm + 0 = x_norm
```

So the network starts fully normalized, then **learns to adjust** the scale and shift during training.

> Normalization removes uncontrolled variation.
> Gamma and beta restore controlled flexibility.

---

## 5. Why LayerNorm Instead of BatchNorm?

### BatchNorm

BatchNorm looks **across examples** in the batch:

```text
Batch:
  Sentence 1: [0.2, -1.1, ...]
  Sentence 2: [0.9,  0.3, ...]
  Sentence 3: [1.4, -0.7, ...]
```

It asks:

> "What is the average feature value across this batch?"

**Problems for Transformers:**

* Batch sizes change at training vs inference.
* Sequence lengths vary — tokens can't be compared across sequences.
* At inference time you may only have one example.

---

### LayerNorm

LayerNorm looks **inside one token**:

```text
Token embedding:
[0.2, -1.1, 0.7, 2.3]
```

It asks:

> "How should I normalize this one token's features?"

Therefore it works naturally with variable-length sequences and single-sample inference.

---

## 6. Pre-Norm vs Post-Norm

### Post-Norm (Original Transformer, 2017)

```python
output = LayerNorm(x + sublayer(x))
```

LayerNorm is applied **after** the residual addition.

**Problem:** In very deep networks, the residual path can accumulate large values before normalization — making training unstable.

---

### Pre-Norm (Modern Transformers: GPT-2, LLaMA)

```python
output = x + sublayer(LayerNorm(x))
```

LayerNorm is applied **before** the sublayer.

The input is always normalized before entering Attention or FFN.

This makes training deeper models easier and more stable.

---

## 7. Visual

```text
Token embedding

[8, 10, 12]

        │
        ▼

Find mean → center values

[-2, 0, 2]

        │
        ▼

Find variance → scale values

[-1.22, 0, 1.22]

        │
        ▼

Gamma + Beta

(learned scale and shift — adjust only if needed)

        │
        ▼

Output embedding

(same shape, stable scale)
```

---

# Coding Assignments

## Assignment 0 — Understand Centering

Before any formulas, implement the centering step alone.

Given:

```python
x = [8, 10, 12]
```

Implement:

```python
def center(x):
    pass
```

Expected output:

```
[-2, 0, 2]
```

Purpose: understand **why** subtracting the mean is the first step.

---

## Assignment 1 — Manual Mean and Variance

Given:

```python
x = np.array([2.0, 4.0, 6.0, 8.0])
```

Compute mean and variance manually (without using `np.mean` or `np.var`).

---

## Assignment 2 — Normalize a Vector

Implement:

```python
def normalize(x, eps=1e-5):
    pass
```

Output should have `mean ≈ 0` and `std ≈ 1`.

---

## Assignment 3 — Layer Norm with Gamma and Beta

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    pass
```

Verify that with `gamma=1, beta=0` the output equals `normalize(x)`.

Then test with `gamma=2, beta=3` — the output should scale and shift accordingly.

---

## Assignment 4 — Apply to Sequence

Apply layer norm independently to each token in a sequence.

Input: `(seq_len, d_model)`
Output: `(seq_len, d_model)`

Verify that each token row independently has `mean ≈ 0` and `std ≈ 1`.

---

# Success Criteria

* Understand why inconsistent scale hurts training.
* Understand that mean centering shifts values around zero.
* Understand that dividing by std controls the spread.
* Understand that normalization preserves relative patterns but removes absolute scale.
* Understand that gamma and beta restore learnable flexibility after normalization.
* Implement LayerNorm from scratch.
* Understand Pre-Norm vs Post-Norm and why modern models prefer Pre-Norm.
