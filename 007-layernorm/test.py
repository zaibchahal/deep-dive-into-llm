import numpy as np
from main import center, normalize, layer_norm

print("Running tests for 007-layernorm...")

# Test center: mean of output should be 0
x0 = np.array([8.0, 10.0, 12.0])
c = center(x0)
assert np.allclose(c, [-2.0, 0.0, 2.0], atol=1e-9), f"Expected [-2,0,2], got {c}"
assert abs(np.mean(c)) < 1e-9, f"Mean after centering should be 0, got {np.mean(c)}"

# Test normalize: mean ~0, std ~1
x = np.array([10.0, -3.0, 200.0, 4.0, 50.0])
x_norm = normalize(x)
assert abs(np.mean(x_norm)) < 1e-5, f"Mean should be ~0, got {np.mean(x_norm)}"
assert abs(np.std(x_norm) - 1.0) < 1e-4, f"Std should be ~1, got {np.std(x_norm)}"

# Test layer_norm with gamma=1, beta=0 equals normalize
gamma = np.ones(5)
beta  = np.zeros(5)
out = layer_norm(x, gamma, beta)
assert np.allclose(out, x_norm, atol=1e-5), "layer_norm(gamma=1,beta=0) should equal normalize(x)"

# Test gamma and beta are applied correctly
gamma2 = np.ones(5) * 2
beta2  = np.ones(5) * 3
out2 = layer_norm(x, gamma2, beta2)
expected = 2 * x_norm + 3
assert np.allclose(out2, expected, atol=1e-5), "gamma=2,beta=3 should give 2*x_norm+3"

# Test sequence shape
x_seq   = np.random.randn(4, 8)
gamma_s = np.ones(8)
beta_s  = np.zeros(8)
out_seq = layer_norm(x_seq, gamma_s, beta_s)
assert out_seq.shape == (4, 8), f"Expected shape (4,8), got {out_seq.shape}"

# Test each token in sequence is independently normalized
for i in range(4):
    assert abs(np.mean(out_seq[i])) < 1e-5, f"Token {i} mean should be ~0"
    assert abs(np.std(out_seq[i]) - 1.0) < 1e-4, f"Token {i} std should be ~1"

print("All tests passed.")
