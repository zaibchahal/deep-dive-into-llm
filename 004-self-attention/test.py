import numpy as np
import math
from main import softmax, attention, causal_mask

print("Running tests for 004-self-attention...")

# Test softmax rows sum to 1
x = np.random.randn(3, 4)
out = softmax(x)
assert out.shape == x.shape
assert np.allclose(out.sum(axis=1), 1.0), "Softmax rows must sum to 1"

# Test attention output shape
Q = np.random.randn(4, 8)
K = np.random.randn(4, 8)
V = np.random.randn(4, 8)
out, weights = attention(Q, K, V)
assert out.shape == (4, 8), f"Expected (4,8), got {out.shape}"
assert weights.shape == (4, 4), f"Expected (4,4), got {weights.shape}"
assert np.allclose(weights.sum(axis=1), 1.0), "Attention weights must sum to 1"

# Test causal mask shape and values
mask = causal_mask(4)
assert mask.shape == (4, 4)
assert mask[0, 1] < -1e8, "Upper triangle should be -inf"
assert mask[1, 0] == 0.0, "Lower triangle should be 0"

# Test causal mask prevents attending to future tokens
out_m, weights_m = attention(Q, K, V, mask=mask)
assert weights_m[0, 1] < 1e-6, "Token 0 should not attend to token 1 (future)"
assert weights_m[0, 0] > 0.99, "Token 0 should attend only to itself"

print("All tests passed.")
