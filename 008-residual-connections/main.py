import numpy as np

np.random.seed(42)


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var  = np.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


# --------------------------
# Assignment 1 — Basic Residual
# --------------------------
# Without residual: sublayer completely replaces x.
# With residual:    sublayer adds a small correction to x.
#
# output = x + sublayer(x)
#
# The output stays close to the input.
# The sublayer only needed to say: "change this by a little."

print("=== Assignment 1: Basic Residual Connection ===")

x = np.array([1.0, 2.0, 3.0, 4.0])
sublayer_output = np.array([0.1, -0.2, 0.3, -0.1])

output = x + sublayer_output

print("x (input):          ", x)
print("sublayer(x):        ", sublayer_output)
print("x + sublayer(x):    ", output)
print("Expected:            [1.1, 1.8, 3.3, 3.9]")
print("Output ≈ input — sublayer made a small correction, not a replacement.")


# --------------------------
# Assignment 2 — Residual with a Function
# --------------------------
# The sublayer is any function.
# The residual wrapper handles the "add input back" part.

def residual_connection(x, sublayer_fn):
    return x + sublayer_fn(x)


print("\n=== Assignment 2: Residual with a Function ===")

def small_correction(x):
    return x * 0.1   # 10% adjustment

x_vec = np.array([1.0, 2.0, 3.0, 4.0])
result = residual_connection(x_vec, small_correction)
print("Input:         ", x_vec)
print("Correction:    ", small_correction(x_vec))
print("After residual:", result)
print("Output stays close to input.")


# --------------------------
# Assignment 3 — Compare Gradient Flow With and Without Shortcut
# --------------------------
# Without shortcut: gradient is multiplied by a small factor at every layer.
# With shortcut:    gradient has a direct path — shortcut adds the gradient back.
#
# Note: in real Transformers gradients are not always literally ≥1.
# The shortcut creates a DIRECT PATH that prevents complete vanishing.
# Here we simulate the principle.

def simulate_gradient(n_layers, use_residual, factor):
    gradient = 1.0
    for _ in range(n_layers):
        if use_residual:
            gradient = gradient * factor + gradient   # shortcut adds gradient back
        else:
            gradient = gradient * factor              # no shortcut
    return gradient


print("\n=== Assignment 3: Gradient Flow Comparison ===")

n_layers = 20

g_no_residual  = simulate_gradient(n_layers, use_residual=False, factor=0.5)
g_with_residual = simulate_gradient(n_layers, use_residual=True,  factor=0.05)

print(f"After {n_layers} layers:")
print(f"  Without shortcut (×0.5 each layer): {g_no_residual:.8f}  ← nearly zero")
print(f"  With shortcut    (×0.05 + shortcut):{g_with_residual:.4f}  ← survives")
print()
print("Without the shortcut the gradient vanishes.")
print("With the shortcut there is always a direct path backward.")


# --------------------------
# Assignment 4 — Combine with Layer Norm
# --------------------------
# Pre-norm residual: normalize first, then pass through sublayer, then add back.
#
#   output = x + sublayer(LayerNorm(x))
#
# LayerNorm stabilizes the input before each sublayer.
# The residual ensures information is never lost.

def pre_norm_residual(x, sublayer_fn, gamma, beta):
    return x + sublayer_fn(layer_norm(x, gamma, beta))


print("\n=== Assignment 4: Pre-Norm Residual ===")

d_model = 8
x_seq = np.random.randn(4, d_model)
gamma = np.ones(d_model)
beta  = np.zeros(d_model)

def dummy_sublayer(x):
    return x * 0.1

output = pre_norm_residual(x_seq, dummy_sublayer, gamma, beta)

print("Input shape: ", x_seq.shape)
print("Output shape:", output.shape)
print("Shapes match:", x_seq.shape == output.shape)
print("\nFirst token input: ", np.round(x_seq[0], 4))
print("First token output:", np.round(output[0], 4))
print("Output ≈ input — small correction applied on top of original.")
