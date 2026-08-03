import numpy as np

np.random.seed(42)


# --------------------------
# Assignment 0 — Understand Centering
# --------------------------
# Before any formulas: understand why we subtract the mean.
# Centering shifts all values so they revolve around zero.
# It removes the absolute offset and keeps only the relative pattern.

def center(x):
    x = np.array(x, dtype=float)
    return x - np.mean(x)


print("=== Assignment 0: Centering ===")
x0 = np.array([8.0, 10.0, 12.0])
centered = center(x0)
print("Input:   ", x0)
print("Centered:", centered)
print("Mean after centering:", round(np.mean(centered), 6), " ← should be 0")


# --------------------------
# Assignment 1 — Manual Mean and Variance
# --------------------------
# Before using numpy, compute these by hand to understand what they measure.
# Mean   → where the values are centered
# Variance → how spread out they are from the mean

print("\n=== Assignment 1: Manual Mean and Variance ===")

x = np.array([2.0, 4.0, 6.0, 8.0])

mean_manual = sum(x) / len(x)
variance_manual = sum((xi - mean_manual) ** 2 for xi in x) / len(x)

print("x:", x)
print("Mean (manual):", mean_manual,    "  numpy:", np.mean(x))
print("Variance (manual):", variance_manual, "  numpy:", np.var(x))


# --------------------------
# Assignment 2 — Normalize a Vector
# --------------------------
# After centering (subtract mean) and scaling (divide by std),
# the vector has mean≈0 and std≈1.
# The relative pattern is preserved; the absolute scale is removed.

def normalize(x, eps=1e-5):
    mean = np.mean(x)
    var = np.var(x)
    return (x - mean) / np.sqrt(var + eps)


print("\n=== Assignment 2: Normalize ===")
x_test = np.array([10.0, -3.0, 200.0, 4.0])
x_norm = normalize(x_test)
print("Input:       ", x_test)
print("Normalized:  ", np.round(x_norm, 4))
print("Mean after norm: ", round(np.mean(x_norm), 6), " ← should be ~0")
print("Std  after norm: ", round(np.std(x_norm),  6), " ← should be ~1")


# --------------------------
# Assignment 3 — Layer Norm with Gamma and Beta
# --------------------------
# After normalization the model has no control over scale.
# Gamma (scale) and beta (shift) restore learnable flexibility.
#
#   output = gamma * x_norm + beta
#
# Initialized to gamma=1, beta=0 → output equals x_norm at the start.
# The model learns the best gamma and beta during training.

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var  = np.var(x,  axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


print("\n=== Assignment 3: Layer Norm with Gamma and Beta ===")

d_model = 8
gamma = np.ones(d_model)
beta  = np.zeros(d_model)

x_vec = np.random.randn(d_model) * 100   # deliberately large values
output = layer_norm(x_vec, gamma, beta)

print("Input (large scale):", np.round(x_vec, 2))
print("Output (normalized): ", np.round(output, 4))
print("Output mean:", round(np.mean(output), 6), " ← ~0")
print("Output std: ", round(np.std(output),  6), " ← ~1")

# Show that gamma/beta let the model adjust
gamma2 = np.ones(d_model) * 2
beta2  = np.ones(d_model) * 3
output2 = layer_norm(x_vec, gamma2, beta2)
print("\nWith gamma=2, beta=3:")
print("Output mean:", round(np.mean(output2), 6), " ← ~3")
print("Output std: ", round(np.std(output2),  6), " ← ~2")


# --------------------------
# Assignment 4 — Apply to Sequence
# --------------------------
# LayerNorm is applied independently to each token.
# No information flows between tokens — same per-token independence as the FFN.
# Each token's vector is normalized on its own.

print("\n=== Assignment 4: Apply Layer Norm to Sequence ===")

seq_len = 5
x_seq = np.random.randn(seq_len, d_model) * 50   # large, inconsistent scale

gamma_seq = np.ones(d_model)
beta_seq  = np.zeros(d_model)

output_seq = layer_norm(x_seq, gamma_seq, beta_seq)

print("Input shape: ", x_seq.shape)
print("Output shape:", output_seq.shape)

print("\nPer-token mean and std after LayerNorm:")
for i in range(seq_len):
    m = np.mean(output_seq[i])
    s = np.std(output_seq[i])
    print(f"  Token {i}: mean={m:.6f}  std={s:.6f}")
