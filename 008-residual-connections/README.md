# 008 — Residual Connections

## Goal

By the end of this module, you should be able to answer:

* What is a residual connection (skip connection)?
* Why does a Transformer block modify an existing representation instead of replacing it completely?
* Why do deep networks suffer from vanishing gradients?
* How do residuals solve both the forward pass problem and the gradient problem?
* Where in the Transformer are residual connections used?

---

# Theory

## 1. The Problem: Deep Networks Can Destroy Information

A Transformer has many stacked blocks:

```text
Embedding
    ↓
Block 1
    ↓
Block 2
    ↓
Block 3
    ↓
  ...
    ↓
Block N
```

Each block transforms the representation. But here is the problem:

> A later block can accidentally destroy useful information learned by earlier blocks.

Example:

The input to a block:

```text
x = [0.8, 0.2, 0.5]
```

The sublayer produces:

```text
sublayer(x) = [-5, 10, 3]
```

This completely replaces the original representation.
Whatever `x` had encoded — context, positional signal, earlier patterns — is gone.

---

## 2. The Solution: Add Instead of Replace

Instead of replacing the representation, add a **correction** on top of it:

```python
output = sublayer(x) + x
```

Without residual — the sublayer must rebuild everything:

```text
x
 ↓
Sublayer
 ↓
new representation   ← original x is gone
```

With residual — the sublayer only learns what to change:

```text
x
 ├──────────────────┐
 ↓                  │
Sublayer            │  shortcut
 ↓                  │
learned correction  │
         +──────────┘

output = original + correction
```

The sublayer only needs to answer:

> "What should I change about this representation?"

Not:

> "How do I rebuild everything from scratch?"

This is a much easier optimization problem.

---

## 3. Why Gradients Vanish Without Shortcuts

The forward pass problem (destroying information) has a mirror image during training: **vanishing gradients**.

During backpropagation, the gradient at early layers is multiplied through every layer in between:

```
gradient at layer 1 = gradient at layer N × (product of all derivatives)
```

With many layers, each slightly less than 1:

```
0.9 × 0.9 × 0.9 × ... × 0.9   (50 layers)
= 0.005
```

The gradient vanishes. Early layers learn almost nothing.

Think of it as passing a message backward through many people:

```text
Person 100 → Person 99 → Person 98 → ... → Person 1
```

Every person slightly weakens the message.
After many steps, almost nothing reaches Person 1.

---

## 4. Residual Creates a Gradient Highway

The shortcut path is:

```text
x ──────────────────→ output
```

No multiplication. No intermediate layers.

During backprop, the gradient through the residual path is:

```
gradient = gradient_through_sublayer + 1
```

The `+1` comes directly from the shortcut. It means the gradient always has a direct route backward — even if the sublayer gradient is tiny.

The shortcut path is the **gradient highway**: it bypasses every intermediate multiplication.

---

## 5. Origin

Introduced in **ResNet** (He et al., 2015) for deep image classification.

Enabled training of 152+ layer networks.

> ResNet showed that very deep networks work better when layers learn **corrections** instead of completely new representations.

Adopted by the Transformer (2017) for exactly the same reason.

---

## 6. In the Transformer

Used twice per block:

```text
Input x
    │
    ▼
LayerNorm
    │
    ▼
Attention
    │
    ▼
+x  ◄────────── shortcut
    │
    ▼
LayerNorm
    │
    ▼
FFN
    │
    ▼
+x  ◄────────── shortcut
    │
    ▼
Output
```

As equations (pre-norm style):

```python
x = x + attention(layer_norm(x))   ← residual 1
x = x + ffn(layer_norm(x))         ← residual 2
```

Every block **builds on** the current representation. None replace it.

---

## 7. Intuition

```text
"Start with what you have, then make small corrections."
```

A translator already understands `"The cat"`.

A later block does not need to recreate that meaning.

It only **adds**:

* grammar relationships
* positional context
* inter-token dependencies

Each block says: *"I will improve what is already here — not start over."*

This is exactly what residuals enable.

---

## 8. Visual

```text
x ──────────────────────────────────┐
│                                    │
▼                                    │
Sublayer (Attention or FFN)          │  shortcut
│                                    │
▼                                    │
learned correction                   │
│                                    │
└──────────────── + ─────────────────┘
                  │
                  ▼
          x + learned correction
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

Expected:

```
[1.1, 1.8, 3.3, 3.9]
```

Notice how close the output is to the input — the sublayer only made a small correction.

---

## Assignment 2 — Residual with a Function

```python
def residual_connection(x, sublayer_fn):
    return x + sublayer_fn(x)
```

Use any sublayer function (e.g., scale by 0.1). Verify the output stays close to the input.

---

## Assignment 3 — Compare Gradient Flow With and Without Shortcut

Simulate gradient flow through many layers.

Without shortcut — multiply a small factor at each layer:

```python
gradient = 1.0
for each layer:
    gradient = gradient * 0.5
```

With shortcut — add the gradient back at each layer:

```python
gradient = 1.0
for each layer:
    gradient = gradient * 0.05 + gradient
```

Print both final gradients and observe the difference.

> Note: in real Transformers gradients are not literally always ≥1. The shortcut creates a **direct path** that prevents complete vanishing, not a hard floor.

---

## Assignment 4 — Combine with Layer Norm

Implement pre-norm residual:

```python
def pre_norm_residual(x, sublayer_fn, gamma, beta):
    return x + sublayer_fn(layer_norm(x, gamma, beta))
```

---

# Success Criteria

* Understand that residuals preserve existing information — sublayers add corrections, not replacements.
* Understand that without residuals, deep layers can destroy earlier representations.
* Understand that residuals create a direct gradient path through the network.
* Understand why gradients vanish in deep networks without shortcuts.
* Know that residuals are used around both Attention and FFN in each Transformer block.
* Implement residual connection and pre-norm residual from scratch.
