import numpy as np
from main import forward, compute_loss, softmax_2d, vocab_size, embedding, W_lm

print("Running tests for 013-train-tiny-gpt...")

# Test forward pass shape
token_ids = [0, 1, 2, 3]
logits = forward(token_ids)
assert logits.shape == (4, vocab_size), f"Expected (4,{vocab_size}), got {logits.shape}"

# Test softmax rows sum to 1
probs = softmax_2d(logits)
assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

# Test loss is positive
loss = compute_loss(logits, [1, 2, 3, 0])
assert loss > 0, "Loss must be positive"

# Test loss for perfect prediction is near 0
perfect_logits = np.zeros((1, vocab_size))
perfect_logits[0, 5] = 100.0
perfect_loss = compute_loss(perfect_logits, [5])
assert perfect_loss < 0.01, f"Perfect prediction should have near-zero loss, got {perfect_loss}"

print("All tests passed.")
