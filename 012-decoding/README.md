# 012 — Decoding

## Goal

By the end of this module, you should be able to answer:

* How does a model generate text token by token?
* What is greedy decoding?
* What is temperature?
* What are top-k and top-p sampling?
* When do you use each strategy?

---

# Theory

## 1. The Generation Loop

After training, we generate text by:

1. Feed a prompt (token IDs)
2. Run the model → get logits for the last position
3. Pick the next token
4. Append it to the sequence
5. Repeat from step 2

This is **autoregressive generation**.

---

## 2. Greedy Decoding

Always pick the **most likely** token:

```
next_token = argmax(probs)
```

Fast and deterministic.

Problem: repetitive, boring outputs.

---

## 3. Temperature Sampling

Divide logits by temperature `T` before softmax:

```
probs = softmax(logits / T)
```

Effects:

| T | Effect |
|---|--------|
| T < 1 | Sharper distribution → more deterministic |
| T = 1 | Standard distribution |
| T > 1 | Flatter distribution → more random |

---

## 4. Top-K Sampling

Keep only the top K tokens.

Set all other probabilities to 0.

Renormalize.

Sample from the reduced distribution.

```
k=50 is common in practice
```

---

## 5. Top-P (Nucleus) Sampling

Sort tokens by probability (highest first).

Keep the smallest set whose cumulative probability ≥ p.

Sample from that set.

```
p=0.9 is common in practice
```

Top-P adapts the number of candidates.

---

## 6. Comparison

| Method | Deterministic | Quality | Diversity |
|--------|--------------|---------|-----------|
| Greedy | Yes | OK | Low |
| Temperature | No | Good | High |
| Top-K | No | Better | Good |
| Top-P | No | Best | Adaptive |

GPT uses top-p + temperature in practice.

---

## 7. Stop Condition

Generation stops when:

* `<EOS>` (end-of-sequence) token is generated
* Max token limit is reached

---

# Coding Assignments

## Assignment 1 — Greedy Decoding

```python
def greedy_decode(logits):
    return np.argmax(logits)
```

---

## Assignment 2 — Temperature Sampling

```python
def temperature_sample(logits, temperature=1.0):
    pass
```

---

## Assignment 3 — Top-K Sampling

```python
def top_k_sample(logits, k=5):
    pass
```

---

## Assignment 4 — Top-P (Nucleus) Sampling

```python
def top_p_sample(logits, p=0.9):
    pass
```

---

## Assignment 5 — Generation Loop

```python
def generate(prompt_ids, model_fn, max_new_tokens=20, strategy="greedy"):
    pass
```

Use a simple model function that returns logits.

---

# Success Criteria

* Implement all four decoding strategies
* Understand how temperature affects output
* Build an autoregressive generation loop
