import numpy as np
from main import residual_connection, pre_norm_residual

print("Running tests for 008-residual-connections...")

# Test basic residual: output = input + sublayer
x = np.array([1.0, 2.0, 3.0, 4.0])
identity = lambda v: np.zeros_like(v)
out = residual_connection(x, identity)
assert np.allclose(out, x), "With zero sublayer, output should equal input"

# Test residual adds correctly
add_one = lambda v: np.ones_like(v)
out2 = residual_connection(x, add_one)
assert np.allclose(out2, x + 1.0)

# Test pre_norm_residual shape
d_model = 8
x_seq = np.random.randn(5, d_model)
gamma = np.ones(d_model)
beta = np.zeros(d_model)
identity_fn = lambda v: np.zeros_like(v)
out3 = pre_norm_residual(x_seq, identity_fn, gamma, beta)
assert out3.shape == x_seq.shape
assert np.allclose(out3, x_seq), "With zero sublayer, pre-norm residual should return input"

print("All tests passed.")
