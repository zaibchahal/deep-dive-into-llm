import numpy as np

np.random.seed(42)


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


# --------------------------
# Assignment 1 — Basic Residual
# --------------------------

print("=== Assignment 1: Basic Residual Connection ===")

x = np.array([1.0, 2.0, 3.0, 4.0])
sublayer_output = np.array([0.1, -0.2, 0.3, -0.1])

output = x + sublayer_output

print("x (input):", x)
print("sublayer(x):", sublayer_output)
print("x + sublayer(x):", output)
print("The output is close to the input — small correction applied.")


# --------------------------
# Assignment 2 — Residual with a Function
# --------------------------

def residual_connection(x, sublayer_fn):
    return x + sublayer_fn(x)


print("\n=== Assignment 2: Residual with a Function ===")

def small_perturbation(x):
    return x * 0.1

x_vec = np.array([1.0, 2.0, 3.0, 4.0])
result = residual_connection(x_vec, small_perturbation)
print("Input:", x_vec)
print("After residual:", result)


# --------------------------
# Assignment 3 — Gradient Flow Demo
# --------------------------

print("\n=== Assignment 3: Gradient Flow Demo ===")

def simulate_gradient(n_layers, use_residual=False, factor=0.5):
    gradient = 1.0
    for layer in range(n_layers):
        if use_residual:
            gradient = gradient * factor + gradient
        else:
            gradient = gradient * factor
    return gradient


n_layers = 20

gradient_no_residual = simulate_gradient(n_layers, use_residual=False, factor=0.5)
gradient_with_residual = simulate_gradient(n_layers, use_residual=True, factor=0.05)

print(f"After {n_layers} layers:")
print(f"  Without residual (×0.5 each layer): {gradient_no_residual:.8f}")
print(f"  With residual (x + 0.05*sublayer):  {gradient_with_residual:.4f}")
print("Residual prevents gradient from going to zero.")


# --------------------------
# Assignment 4 — Combine with Layer Norm
# --------------------------

def pre_norm_residual(x, sublayer_fn, gamma, beta):
    return x + sublayer_fn(layer_norm(x, gamma, beta))


print("\n=== Assignment 4: Pre-Norm Residual ===")

d_model = 8
x_seq = np.random.randn(4, d_model)
gamma = np.ones(d_model)
beta = np.zeros(d_model)

def dummy_sublayer(x):
    return x * 0.1

output = pre_norm_residual(x_seq, dummy_sublayer, gamma, beta)

print("Input shape:", x_seq.shape)
print("Output shape:", output.shape)
print("Input and output shapes match:", x_seq.shape == output.shape)
print("\nFirst token input:", np.round(x_seq[0], 4))
print("First token output:", np.round(output[0], 4))
print("Output is close to input (residual adds small correction).")
