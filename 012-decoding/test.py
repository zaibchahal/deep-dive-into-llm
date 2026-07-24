import numpy as np
from main import greedy_decode, temperature_sample, top_k_sample, top_p_sample

print("Running tests for 012-decoding...")

np.random.seed(0)

# Test greedy always picks argmax
logits = np.array([0.1, 0.5, 3.0, 0.2])
assert greedy_decode(logits) == 2

# Test temperature=0.001 behaves like greedy
tokens = [temperature_sample(logits, temperature=0.001) for _ in range(100)]
assert all(t == 2 for t in tokens), "Near-zero temperature should always pick argmax"

# Test top_k only returns top k indices
logits2 = np.array([0.1, 5.0, 0.2, 4.0, 0.05])
results = set(top_k_sample(logits2, k=2) for _ in range(200))
assert results.issubset({1, 3}), f"top_k=2 should only return tokens 1 or 3, got {results}"

# Test top_p returns a valid token index
logits3 = np.array([1.0, 2.0, 0.5, 0.1, 3.0])
for _ in range(50):
    t = top_p_sample(logits3, p=0.9)
    assert 0 <= t < len(logits3)

print("All tests passed.")
