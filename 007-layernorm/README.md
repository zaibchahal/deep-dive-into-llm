# 007 — Layer Normalization

## Goal

By the end of this module, you should be able to answer:

* What is Layer Normalization?
* Why do Transformers need normalization?
* What is the difference between Batch Norm and Layer Norm?
* What are gamma and beta parameters?

---

# Theory

## 1. The Problem: Training Instability

During training, activations can grow very large or very small.

Example:

```
After FFN:  [-230, 450, -18, 12000]
```

Large values cause:

* Gradient explosion
* Saturated activations
* Slow or failed training

---

## 2. Normalization

We want each vector to have:

```
mean  ≈ 0
variance ≈ 1
```

This keeps activations in a healthy range.

---

## 3. Layer Norm Formula

For a single vector `x` of length `d`:

```
μ  = mean(x)
σ² = variance(x)

x_norm = (x - μ) / sqrt(σ² + ε)

output = gamma * x_norm + beta
```

Where:

```
ε     = small constant for numerical stability (e.g., 1e-5)
gamma = learned scale parameter (shape: d)
beta  = learned shift parameter (shape: d)
```

Initially:

```
gamma = ones
beta  = zeros
```

---

## 4. Layer Norm vs Batch Norm

| | Batch Norm | Layer Norm |
|-|------------|------------|
| Normalizes over | batch dimension | feature dimension |
| Depends on batch size | Yes | No |
| Works at test time | needs running stats | yes, per-sample |
| Used in | CNNs | Transformers |

Transformers use **Layer Norm** because sequences have variable lengths and batch normalization doesn't work well.

---

## 5. Pre-Norm vs Post-Norm

Original Transformer (2017): **Post-Norm**

```
output = LayerNorm(x + sublayer(x))
```

Modern models (GPT-2+): **Pre-Norm**

```
output = x + sublayer(LayerNorm(x))
```

Pre-norm is more stable during training.

---

## 6. Visual

```
x = [10, -3, 200, 4]

μ = (10 + -3 + 200 + 4) / 4 = 52.75
σ² = variance

x_norm = (x - μ) / √σ²
       = [-0.98, -1.22, 3.22, -1.02]

output = gamma * x_norm + beta
```

---

# Coding Assignments

## Assignment 1 — Manual Mean and Variance

Given:

```python
x = np.array([2.0, 4.0, 6.0, 8.0])
```

Compute mean and variance manually (without numpy functions).

---

## Assignment 2 — Normalize a Vector

Implement:

```python
def normalize(x, eps=1e-5):
    pass
```

Output should have mean ≈ 0, std ≈ 1.

---

## Assignment 3 — Layer Norm with Gamma and Beta

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    pass
```

---

## Assignment 4 — Apply to Sequence

Apply layer norm independently to each token in a sequence.

Input: `(seq_len, d_model)`
Output: `(seq_len, d_model)`

---

# Success Criteria

* Implement layer norm from scratch
* Understand gamma and beta
* Know why layer norm is preferred over batch norm for Transformers
* Know pre-norm vs post-norm
