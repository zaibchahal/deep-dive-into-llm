import numpy as np
from main import multi_head_attention

print("Running tests for 005-multi-head-attention...")

np.random.seed(0)

# Test output shape matches input shape
x = np.random.randn(5, 8)
out = multi_head_attention(x, h=4)
assert out.shape == (5, 8), f"Expected (5, 8), got {out.shape}"

# Test different inputs produce different outputs
x2 = np.random.randn(5, 8)
out2 = multi_head_attention(x2, h=4)
assert not np.allclose(out, out2), "Different inputs should give different outputs"

# Test works with h=2
x3 = np.random.randn(3, 8)
out3 = multi_head_attention(x3, h=2)
assert out3.shape == (3, 8)

print("All tests passed.")
