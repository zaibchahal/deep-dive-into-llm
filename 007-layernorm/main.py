import numpy as np

np.random.seed(42)


# --------------------------
# Assignment 1 — Manual Mean and Variance
# --------------------------

print("=== Assignment 1: Manual Mean and Variance ===")

x = np.array([2.0, 4.0, 6.0, 8.0])

mean_manual = sum(x) / len(x)

variance_manual = sum((xi - mean_manual) ** 2 for xi in x) / len(x)

print("x:", x)
print("Mean (manual):", mean_manual)
print("Mean (numpy):", np.mean(x))
print("Variance (manual):", variance_manual)
print("Variance (numpy):", np.var(x))


# --------------------------
# Assignment 2 — Normalize a Vector
# --------------------------

def normalize(x, eps=1e-5):
    mean = np.mean(x)
    var = np.var(x)
    return (x - mean) / np.sqrt(var + eps)


print("\n=== Assignment 2: Normalize ===")
x_test = np.array([10.0, -3.0, 200.0, 4.0])
x_norm = normalize(x_test)
print("Input:", x_test)
print("Normalized:", np.round(x_norm, 4))
print("Mean after norm:", round(np.mean(x_norm), 6))
print("Std after norm:", round(np.std(x_norm), 6))


# --------------------------
# Assignment 3 — Layer Norm with Gamma and Beta
# --------------------------

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


print("\n=== Assignment 3: Layer Norm with Gamma and Beta ===")

d_model = 8
gamma = np.ones(d_model)
beta = np.zeros(d_model)

x_vec = np.random.randn(d_model) * 100
output = layer_norm(x_vec, gamma, beta)

print("Input (large values):", np.round(x_vec, 2))
print("Output (normalized):", np.round(output, 4))
print("Output mean:", round(np.mean(output), 6))
print("Output std:", round(np.std(output), 6))


# --------------------------
# Assignment 4 — Apply to Sequence
# --------------------------

print("\n=== Assignment 4: Apply Layer Norm to Sequence ===")

seq_len = 5
x_seq = np.random.randn(seq_len, d_model) * 50

gamma_seq = np.ones(d_model)
beta_seq = np.zeros(d_model)

output_seq = layer_norm(x_seq, gamma_seq, beta_seq)

print("Input shape:", x_seq.shape)
print("Output shape:", output_seq.shape)

print("\nPer-token mean and std after layer norm:")
for i in range(seq_len):
    print(f"  Token {i}: mean={np.mean(output_seq[i]):.6f}, std={np.std(output_seq[i]):.6f}")
