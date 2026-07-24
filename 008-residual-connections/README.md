# 008 — Residual Connections

## Goal

By the end of this module, you should be able to answer:

* What is a residual connection (skip connection)?
* Why do deep networks suffer from vanishing gradients?
* How do residuals solve this?
* Where in the Transformer are residual connections used?

---

# Theory

## 1. The Problem: Vanishing Gradients in Deep Networks

Deep networks can have many layers.

During backpropagation:

```
gradient at layer 1 = gradient at layer N × (product of derivatives)
```

With many layers:

```
0.9 × 0.9 × 0.9 × ... × 0.9  (50 layers)

= 0.005
```

The gradient vanishes.

Early layers learn almost nothing.

---

## 2. The Solution: Residual Connections (Skip Connections)

Add the **input** directly to the **output** of a sublayer:

```
output = sublayer(x) + x
```

This creates a **shortcut** for the gradient.

During backprop:

```
gradient = gradient_through_sublayer + 1
```

The `+ 1` ensures the gradient never completely vanishes.

---

## 3. Origin

Introduced in **ResNet** (He et al., 2015) for deep image classification.

Enabled training of 152+ layer networks.

Adopted by Transformer (2017).

---

## 4. In the Transformer

Used twice per block:

```
After self-attention:
  x = x + attention(x)

After feed-forward:
  x = x + ffn(x)
```

With layer norm (pre-norm style):

```
x = x + attention(layer_norm(x))
x = x + ffn(layer_norm(x))
```

---

## 5. Intuition

Think of residuals as:

```
"Start with what you have, then make small corrections."
```

The sublayer only needs to learn the **difference** from the input.

This is called a **residual**.

---

## 6. Visual

```
x ─────────────────────────────┐
│                               │
↓                               │
Sublayer (attention or FFN)    │
│                               │
↓                               │
sublayer(x)                     │
│                               │
└───────────── + ───────────────┘
               │
               ↓
       x + sublayer(x)
```

---

# Coding Assignments

## Assignment 1 — Basic Residual

Given:

```python
x = np.array([1.0, 2.0, 3.0, 4.0])
sublayer_output = np.array([0.1, -0.2, 0.3, -0.1])
```

Compute:

```python
output = x + sublayer_output
```

---

## Assignment 2 — Residual with a Function

```python
def residual_connection(x, sublayer_fn):
    return x + sublayer_fn(x)
```

Use any sublayer function (e.g., add noise, scale).

---

## Assignment 3 — Gradient Flow Demo

Show that with residual connection, the gradient is at least 1.

Without residual:

```
gradient ≈ 0.0001 (after many multiplications)
```

With residual:

```
gradient ≥ 1
```

Simulate with a loop.

---

## Assignment 4 — Combine with Layer Norm

Implement pre-norm residual:

```python
def pre_norm_residual(x, sublayer_fn, gamma, beta):
    return x + sublayer_fn(layer_norm(x, gamma, beta))
```

---

# Success Criteria

* Understand why gradients vanish
* Know what a skip connection is
* Implement residual + layer norm
* Know where residuals appear in a Transformer block
