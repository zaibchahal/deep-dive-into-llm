# 011 — Next Token Prediction

## Goal

By the end of this module, you should be able to answer:

* How does a language model predict the next token?
* What is the LM Head?
* What is cross-entropy loss?
* How does teacher forcing work during training?

---

# Theory

## 1. The LM Head

After the Transformer stack:

```
Output: (seq_len, d_model)
```

We need to predict the **next token** — one of `vocab_size` options.

The Language Model Head:

```
Linear layer: d_model → vocab_size
```

```
Output: (seq_len, vocab_size)
```

Each row is a score (logit) for every token in the vocabulary.

---

## 2. Logits → Probabilities

Apply Softmax:

```
probs = softmax(logits)
```

Each row now sums to 1.

```
probs[i][j] = probability that token j comes after token i
```

---

## 3. Full Forward Pass

```
Token IDs
   ↓
Embedding + Positional Encoding
   ↓
Transformer Stack
   ↓
Final LayerNorm
   ↓
LM Head (Linear: d_model → vocab_size)
   ↓
Logits  (seq_len, vocab_size)
   ↓
Softmax
   ↓
Probabilities
   ↓
Predicted next token = argmax(probs)
```

---

## 4. Cross-Entropy Loss

During training, the model predicts the next token.

We know the correct next token.

Loss = how wrong the prediction was.

Formula:

```
loss = -log( predicted_prob_of_correct_token )
```

Example:

```
Correct next token: "cat" (id=5)

Model prediction:
  "the" → 0.05
  "a"   → 0.10
  "cat" → 0.60
  ...

loss = -log(0.60) = 0.51
```

Lower loss = better predictions.

---

## 5. Teacher Forcing

During training, we don't use the model's own predictions.

We use the **ground truth** sequence as input:

```
Input:  "The cat sat"
Target: "cat sat on"
```

The model is given the correct previous tokens at each step.

This is teacher forcing — it stabilizes training.

---

## 6. Shift by One

The prediction target is shifted by 1:

```
Input tokens:   [The, cat, sat, on, the]
                  ↓     ↓    ↓   ↓    ↓
Target tokens:  [cat, sat,  on, the, mat]
```

Token 0 predicts Token 1.
Token 1 predicts Token 2.
etc.

---

# Coding Assignments

## Assignment 1 — LM Head

Create a linear layer:

```python
W_lm = np.random.randn(d_model, vocab_size)
logits = context_vectors @ W_lm
```

---

## Assignment 2 — Top Prediction

```python
next_token_id = np.argmax(probs[-1])
```

The last token's row predicts what comes next.

---

## Assignment 3 — Cross-Entropy Loss

Implement:

```python
def cross_entropy(logits, targets):
    pass
```

Where `targets` is a list of correct token IDs.

---

## Assignment 4 — Loss on a Batch

Compute the average loss over a sequence.

---

# Success Criteria

* Know what the LM Head does
* Understand logits vs probabilities
* Implement cross-entropy loss
* Understand teacher forcing and the shift-by-one target
