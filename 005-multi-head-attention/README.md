# 005 — Multi-Head Attention

## Goal

By the end of this module, you should be able to answer:

* What is multi-head attention?
* Why run multiple attention heads in parallel?
* How are the heads combined?
* What does each head learn?

---

# Theory

## 1. Problem with Single-Head Attention

Single attention head computes one set of attention weights.

It can only focus on one type of relationship at a time.

Example:

```
"The animal didn't cross the street because it was too tired"
```

One head might learn:

```
"it" → "animal"   (coreference)
```

But the model might also need to track:

```
"cross" → "street"   (verb-object)
"tired" → "animal"   (property)
```

One attention head can't do all of this at once.

---

## 2. Multi-Head Attention

Run `h` attention heads in parallel.

Each head has its own Q, K, V projection matrices.

Each head learns a **different aspect** of the input.

```
Head 1: Q1, K1, V1 → output1
Head 2: Q2, K2, V2 → output2
...
Head h: Qh, Kh, Vh → outputh
```

Concatenate all outputs:

```
concat(output1, output2, ..., outputh)
```

Project back to d_model:

```
final = concat(...) @ W_O
```

---

## 3. Dimension Split

We split `d_model` across heads:

```
d_k = d_model / h
```

Example:

```
d_model = 512
h       = 8
d_k     = 64   (per head)
```

Each head uses smaller projections.

Total computation stays the same as single head.

---

## 4. Visual Flow

```
Input: (seq_len, d_model)
          ↓
   Split into h heads
          ↓
Head 1     Head 2  ...  Head h
  ↓          ↓             ↓
Attn 1    Attn 2   ...  Attn h
(seq,d_k) (seq,d_k)    (seq,d_k)
          ↓
   Concatenate → (seq, d_model)
          ↓
   Linear W_O → (seq, d_model)
```

---

## 5. Formula

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

where head_i = Attention(Q @ W_Qi, K @ W_Ki, V @ W_Vi)
```

---

# Coding Assignments

## Assignment 1 — Single Head (Review)

Implement one attention head with `d_k = d_model / h`.

---

## Assignment 2 — Multiple Heads in Loop

For `h=4` heads:

```python
for i in range(h):
    # create Q_i, K_i, V_i
    # compute attention
    # collect output_i
```

---

## Assignment 3 — Concatenate and Project

```python
concat = np.concatenate(all_head_outputs, axis=-1)
output = concat @ W_O
```

---

## Assignment 4 — Full Multi-Head Attention Function

```python
def multi_head_attention(x, h=4):
    pass
```

Input: `(seq_len, d_model)`
Output: `(seq_len, d_model)`

---

# Success Criteria

* Know why one head is not enough
* Understand how dimension is split across heads
* Implement multi-head attention from scratch
* Know the shape at each step
