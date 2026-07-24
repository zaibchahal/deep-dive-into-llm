import numpy as np
from main import normalize, layer_norm

print("Running tests for 007-layernorm...")

# Test normalize output has mean ~0 and std ~1
x = np.array([10.0, -3.0, 200.0, 4.0, 50.0])
x_norm = normalize(x)
assert abs(np.mean(x_norm)) < 1e-5, f"Mean should be ~0, got {np.mean(x_norm)}"
assert abs(np.std(x_norm) - 1.0) < 1e-4, f"Std should be ~1, got {np.std(x_norm)}"

# Test layer_norm with gamma=1, beta=0 equals normalize
gamma = np.ones(5)
beta = np.zeros(5)
out = layer_norm(x, gamma, beta)
assert np.allclose(out, x_norm, atol=1e-5)

# Test gamma and beta are applied
gamma2 = np.ones(5) * 2
beta2 = np.ones(5) * 3
out2 = layer_norm(x, gamma2, beta2)
expected = 2 * x_norm + 3
assert np.allclose(out2, expected, atol=1e-5)

# Test sequence shape
x_seq = np.random.randn(4, 8)
gamma_s = np.ones(8)
beta_s = np.zeros(8)
out_seq = layer_norm(x_seq, gamma_s, beta_s)
assert out_seq.shape == (4, 8)

print("All tests passed.")
