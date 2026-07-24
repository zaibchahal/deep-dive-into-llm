import numpy as np
from main import cross_entropy, softmax

print("Running tests for 011-next-token-prediction...")

# Test softmax sums to 1
logits = np.random.randn(4, 10)
probs = softmax(logits)
assert np.allclose(probs.sum(axis=1), 1.0)

# Test cross_entropy: perfect prediction gives loss near 0
logits_perfect = np.zeros((3, 5))
targets = [0, 1, 2]
for i, t in enumerate(targets):
    logits_perfect[i, t] = 100.0
loss = cross_entropy(logits_perfect, targets)
assert loss < 0.01, f"Perfect prediction should have near-zero loss, got {loss}"

# Test cross_entropy: wrong prediction gives high loss
logits_wrong = np.zeros((1, 5))
logits_wrong[0, 0] = 100.0
loss_wrong = cross_entropy(logits_wrong, [4])
assert loss_wrong > 4.0, f"Wrong prediction should have high loss, got {loss_wrong}"

# Test argmax picks correct token
logits_test = np.zeros((1, 10))
logits_test[0, 7] = 5.0
probs_test = softmax(logits_test)
assert np.argmax(probs_test[0]) == 7

print("All tests passed.")
