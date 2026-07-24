import numpy as np
from main import relu, gelu, ffn, ffn_sequence

print("Running tests for 006-feed-forward-network...")

# Test relu
x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
r = relu(x)
assert r[0] == 0.0
assert r[1] == 0.0
assert r[2] == 0.0
assert r[3] == 1.0
assert r[4] == 3.0

# Test gelu: positive output for positive input
assert gelu(np.array([1.0]))[0] > 0
# gelu(0) should be 0
assert abs(gelu(np.array([0.0]))[0]) < 1e-9

# Test ffn shape
d_model = 8
d_ff = 32
W1 = np.random.randn(d_model, d_ff)
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model)
b2 = np.zeros(d_model)
x_single = np.random.randn(d_model)
out = ffn(x_single, W1, b1, W2, b2)
assert out.shape == (d_model,)

# Test ffn_sequence shape
x_seq = np.random.randn(5, d_model)
out_seq = ffn_sequence(x_seq, W1, b1, W2, b2)
assert out_seq.shape == (5, d_model)

print("All tests passed.")
