import numpy as np
import math

np.random.seed(42)


# --------------------------
# Assignment 1 — ReLU Activation
# --------------------------

def relu(x):
    return np.maximum(0, x)


print("=== Assignment 1: ReLU ===")
test_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print("Input:", test_values)
print("ReLU:", relu(test_values))


# --------------------------
# Assignment 2 — GELU Activation
# --------------------------

def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


print("\n=== Assignment 2: GELU ===")
print("Input:", test_values)
print("GELU:", np.round(gelu(test_values), 4))
print("Note: GELU is smoother than ReLU (negative values not hard-zeroed)")


# --------------------------
# Assignment 3 — Single FFN Forward Pass
# --------------------------

def ffn(x, W1, b1, W2, b2, activation=gelu):
    hidden = activation(x @ W1 + b1)
    output = hidden @ W2 + b2
    return output


print("\n=== Assignment 3: Single FFN Forward Pass ===")

d_model = 8
d_ff = 32  # 4× expansion

x_single = np.random.randn(d_model)
W1 = np.random.randn(d_model, d_ff)
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model)
b2 = np.zeros(d_model)

output_single = ffn(x_single, W1, b1, W2, b2)
print("Input shape:", x_single.shape)
print("W1 shape (d_model → d_ff):", W1.shape)
print("W2 shape (d_ff → d_model):", W2.shape)
print("Output shape:", output_single.shape)
print("Output:", np.round(output_single, 4))


# --------------------------
# Assignment 4 — Apply FFN to All Tokens
# --------------------------

def ffn_sequence(x_seq, W1, b1, W2, b2, activation=gelu):
    seq_len = x_seq.shape[0]
    outputs = np.zeros_like(x_seq)
    for i in range(seq_len):
        outputs[i] = ffn(x_seq[i], W1, b1, W2, b2, activation)
    return outputs


print("\n=== Assignment 4: Apply FFN to Sequence ===")

seq_len = 5
x_seq = np.random.randn(seq_len, d_model)

output_seq = ffn_sequence(x_seq, W1, b1, W2, b2)
print("Input shape:", x_seq.shape)
print("Output shape:", output_seq.shape)
print("Input shape == Output shape:", x_seq.shape == output_seq.shape)

print("\nFirst token:")
print("  Input:", np.round(x_seq[0], 4))
print("  Output:", np.round(output_seq[0], 4))
