# 015 — LoRA (Low-Rank Adaptation)

## References

* [LoRA Paper — Hu et al. 2021](https://arxiv.org/abs/2106.09685) — The original "LoRA: Low-Rank Adaptation of Large Language Models" paper.
* [Illustrated LoRA — Sebastian Raschka](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — Practical intuition and tips for fine-tuning LLMs with LoRA.
* [HuggingFace PEFT](https://huggingface.co/docs/peft) — The standard library for applying LoRA (and QLoRA) to any HuggingFace model.

---

## Goal

By the end of this module, you should be able to answer:

* What is fine-tuning and why is it expensive?
* What is LoRA?
* How does low-rank decomposition work?
* How many parameters does LoRA add?
* What is rank `r` and `alpha`?
* What is QLoRA and how does it differ from LoRA?
* How do you merge a LoRA adapter back into the base model?
* Which layers should you apply LoRA to?

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

In HuggingFace PEFT, these are called **target modules**:

```python
target_modules = ["q_proj", "v_proj"]          # minimal
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # full attention
```

---

## 9. Merging Adapters

After training, you can **merge** the adapter back into the base weights:

```
W_merged = W + (alpha / r) × (B @ A)
```

Result: a single matrix, **zero inference overhead**.

```python
model = model.merge_and_unload()   # PEFT one-liner
```

The merged model is identical in size to the original — but now fine-tuned.

---

## 10. QLoRA

QLoRA = **4-bit quantized base model** + LoRA adapters on top.

```
Base model weights:  4-bit (frozen, quantized with bitsandbytes)
LoRA A, B matrices:  bf16 (trainable)
```

Why it matters:

| Method       | GPU RAM (7B model) |
| ------------ | ------------------ |
| Full fine-tune | ~112 GB          |
| LoRA (bf16)  | ~16 GB             |
| QLoRA (4-bit)| ~6 GB              |

QLoRA lets you fine-tune a 7B model on a single consumer GPU.

Key addition: **double quantization** + **NF4 data type** (Normal Float 4).

---

## 11. Common Hyperparameters in Practice

| Hyperparameter | Typical Value | Notes |
| -------------- | ------------- | ----- |
| `r`            | 8 or 16       | Higher = more expressive, more params |
| `alpha`        | 16 or 32      | Often set to `2 × r` |
| `dropout`      | 0.05          | Applied to LoRA layers for regularisation |
| `target_modules` | `q_proj, v_proj` | Start minimal, expand if underfitting |
| `bias`         | `none`        | Usually don't train biases |

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

## Assignment 5 — Merge the Adapter

After training A and B, merge them back:

```python
W_merged = W + (alpha / r) * (B @ A)
```

Verify that `W_merged @ x` gives the same output as the LoRA forward pass.

---

## Assignment 6 — Apply LoRA with PEFT

Use HuggingFace PEFT to apply LoRA to a small model (e.g. `gpt2`):

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"])
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

Check how many parameters are trainable vs frozen.

---

## Assignment 7 — Rank Ablation

Train the same LoRA adapter at `r = 1, 4, 8, 32`.

Plot: final loss vs parameter count.

Observe: at what rank does performance plateau?

---

# Success Criteria

* Understand why LoRA uses fewer parameters
* Implement LoRA forward pass from scratch
* Train A and B while W is frozen and confirm loss drops
* Merge adapter back into base weights with zero overhead
* Know what `r`, `alpha`, `target_modules`, and `dropout` do
* Explain the difference between LoRA and QLoRA
* Use PEFT to apply LoRA to a real HuggingFace model
