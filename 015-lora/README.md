# 015 — LoRA (Low-Rank Adaptation)

## Goal

By the end of this module, you should be able to answer:

* What is fine-tuning and why is it expensive?
* What is LoRA?
* How does low-rank decomposition work?
* How many parameters does LoRA add?
* What is rank r and alpha?

---

# Theory

## 1. The Problem: Fine-Tuning is Expensive

GPT-3 has 175 billion parameters.

Fine-tuning = updating all 175B weights.

Requires enormous GPU memory and compute.

---

## 2. Key Insight from LoRA

When fine-tuning, the weight changes are often low-rank.

Original weight matrix:

```
W  shape: (d_out, d_in)   e.g., (4096, 4096) = 16M params
```

The change during fine-tuning:

```
ΔW ≈ small rank matrix
```

We can approximate ΔW as:

```
ΔW = B × A

A: (r, d_in)    small
B: (d_out, r)   small

where r << d_in, d_out
```

---

## 3. LoRA Forward Pass

Original:

```
output = x @ W
```

With LoRA:

```
output = x @ W + x @ A.T @ B.T
```

Which is:

```
output = x @ (W + ΔW)
       = x @ W + x @ A.T @ B.T
```

W is frozen.

Only A and B are trained.

---

## 4. Parameter Count

Original W:

```
d_out × d_in = 4096 × 4096 = 16,777,216 params
```

LoRA A + B (r=16):

```
r × d_in  = 16 × 4096 = 65,536
d_out × r = 4096 × 16 = 65,536
Total: 131,072 params
```

Reduction: 128× fewer trainable parameters!

---

## 5. Rank r

Controls the expressiveness of the adaptation.

```
r = 1    minimum, very few params
r = 8    common default
r = 64   more expressive, more params
r = 256  large
```

---

## 6. Alpha (Scaling)

```
output = x @ W + (alpha / r) × (x @ A.T @ B.T)
```

`alpha` controls the scale of the adaptation.

Common: `alpha = r` (scale = 1) or `alpha = 2r`.

---

## 7. Initialization

```
A: random normal (small)
B: zeros
```

At the start:

```
ΔW = B × A = 0 × A = 0
```

The model starts identical to the pretrained model.

---

## 8. Which Weights Get LoRA?

Applied to attention weight matrices:

```
W_Q, W_K, W_V, W_O
```

Sometimes also FFN weights.

Not applied to layer norms or embeddings.

---

# Coding Assignments

## Assignment 1 — Low-Rank Matrix

Show that a matrix can be approximated by its low-rank factorization:

```python
# Full matrix
W = np.random.randn(100, 100)

# Low rank approximation with rank 5
A = np.random.randn(5, 100)
B = np.random.randn(100, 5)

W_approx = B @ A
```

Compare rank and size.

---

## Assignment 2 — LoRA Layer

```python
class LoRALayer:
    def __init__(self, W, r=4, alpha=1):
        self.W = W          # frozen
        self.A = ...        # trainable
        self.B = ...        # trainable
        self.scale = alpha / r

    def forward(self, x):
        return x @ self.W + self.scale * (x @ self.A.T @ self.B.T)
```

---

## Assignment 3 — Count Parameters

Compare original vs LoRA parameter counts.

---

## Assignment 4 — Train LoRA Adapter

Freeze W.

Train only A and B to minimize a simple loss.

Show that loss decreases.

---

# Success Criteria

* Understand why LoRA uses fewer parameters
* Implement LoRA forward pass
* Train A and B while W is frozen
* Know rank, alpha, and which layers get LoRA
