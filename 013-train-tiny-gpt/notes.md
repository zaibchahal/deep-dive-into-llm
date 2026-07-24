# 013 — Train Tiny GPT: Notes

## The Training Loop

```
for each epoch:
    forward(x) → logits
    loss = cross_entropy(logits, targets)
    grad = d_loss / d_weights
    weights -= lr × grad
```

## Gradient Descent

Move weights in the direction that reduces loss.

Learning rate (lr): how big a step to take.
- Too high: diverges (loss explodes)
- Too low: slow convergence
- Good range: 0.001–0.1

## Loss Curve

Should go down over time.

Flat loss = model is stuck or lr is wrong.
Increasing loss = lr too high.

## Character-Level Model

Simplest possible tokenization: one char = one token.

Vocabulary ≈ 30-60 chars.

Works well for short training texts.

## Why Simplified?

Full GPT training requires:
- Backprop through attention, softmax, layer norm
- GPU / matrix autodiff
- Terabytes of data

Our tiny model: embedding + LM head, character-level, NumPy only.

## What Real Training Looks Like

```python
# PyTorch (the real thing)
optimizer = Adam(model.parameters(), lr=3e-4)
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Next

**Inference** — loading a trained model and generating text efficiently.
