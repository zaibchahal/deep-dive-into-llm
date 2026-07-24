# 013 — Train a Tiny GPT

## Goal

By the end of this module, you should be able to answer:

* What is a training loop?
* What is gradient descent?
* How does backpropagation work conceptually?
* What is a loss curve?
* How does a tiny GPT learn to generate text?

---

# Theory

## 1. What Training Means

Training = finding parameters that minimize the loss.

Loss = how wrong the model's predictions are.

Parameters = all the weights (W_Q, W_K, W_V, W_O, W1, W2, embeddings).

---

## 2. Gradient Descent

```
For each batch of training data:
  1. Forward pass: compute predictions and loss
  2. Backward pass: compute gradients (how to change each weight)
  3. Update weights: move in the direction that reduces loss
```

Update rule:

```
weight = weight - learning_rate × gradient
```

`learning_rate` (lr) controls the step size.

---

## 3. Loss Curve

```
Epoch 1:  loss = 4.2
Epoch 5:  loss = 3.1
Epoch 20: loss = 1.8
Epoch 50: loss = 0.9
```

Loss goes down as the model learns.

---

## 4. Character-Level GPT

For simplicity, train on characters instead of BPE tokens.

Vocabulary = all unique characters in the training text.

Example:

```
Training text: "hello world"

Vocab: {' ':0, 'd':1, 'e':2, 'h':3, 'l':4, 'o':5, 'r':6, 'w':7}

Encoded: [3, 2, 4, 4, 5, 0, 7, 5, 6, 4, 1]
```

---

## 5. Backprop (Conceptual)

Real backprop through a Transformer requires computing derivatives of:

* Softmax
* Matrix multiplications
* Layer norm

We will use **numerical gradient approximation**:

```
grad ≈ (f(x + ε) - f(x - ε)) / (2ε)
```

Or use a simplified single-layer model where gradients are easy to compute.

---

## 6. Why Tiny?

Real GPT training requires:

* Terabytes of text
* Thousands of GPUs
* Weeks of training time

Our tiny GPT:

* Short text (a poem or paragraph)
* Character-level vocabulary (~30-60 chars)
* 1-2 Transformer blocks
* d_model = 32-64
* CPU only
* Trains in seconds/minutes

---

# Coding Assignments

## Assignment 1 — Prepare Data

Load a short text.

Build character vocabulary.

Encode text to token IDs.

Create training pairs (input, target) with a sliding window.

---

## Assignment 2 — Build a Tiny Model

Combine:

* Embedding
* Positional encoding
* 2 Transformer blocks
* LM Head

---

## Assignment 3 — Training Loop (Simplified)

Use a single linear model (embedding + LM head only) to keep gradients simple.

Train for N epochs.

Print loss every 10 epochs.

---

## Assignment 4 — Generate Text

After training, use greedy decoding to generate characters.

Show improvement over epochs.

---

# Success Criteria

* Understand the training loop conceptually
* Build a tiny model from scratch
* See loss decrease over training
* Generate coherent text after training
