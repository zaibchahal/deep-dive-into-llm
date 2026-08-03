import numpy as np
import math

np.random.seed(42)


# --------------------------
# Assignment 0 — One Neuron
# --------------------------
# A neuron is one set of weights that produces one output value.
# output = x[0]*w[0] + x[1]*w[1] + ... + b

def neuron(x, w, b):
    x = np.array(x, dtype=float)
    w = np.array(w, dtype=float)
    return float(x @ w + b)


print("=== Assignment 0: One Neuron ===")
x = [2, 3, 4]
w = [1, -2, 0.5]
b = 1
result = neuron(x, w, b)
# Manual: 2*1 + 3*(-2) + 4*0.5 + 1 = 2 - 6 + 2 + 1 = -1.0
print(f"x={x}  w={w}  b={b}")
print(f"Neuron output: {result}")
print(f"Manual check:  2*1 + 3*(-2) + 4*0.5 + 1 = {2*1 + 3*(-2) + 4*0.5 + 1}")


# --------------------------
# Assignment 1 — ReLU Activation
# --------------------------
# The activation function determines which neurons are active.
# ReLU turns negative values into zero:
#   "You have nothing useful to contribute for this token."

def relu(x):
    return np.maximum(0, x)


print("\n=== Assignment 1: ReLU ===")
test_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print("Input:", test_values)
print("ReLU:", relu(test_values))

# Visual intuition:
print("\nReLU intuition:")
print("  Before:", [-3, 5, -2, 8])
print("  After: ", list(relu(np.array([-3, 5, -2, 8])).astype(int)))
print("  Only neurons with positive outputs stay active.")


# --------------------------
# Assignment 2 — GELU Activation
# --------------------------
# GELU is smoother than ReLU — instead of a hard cutoff at zero,
# it gradually suppresses near-zero values.

def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


print("\n=== Assignment 2: GELU ===")
print("Input:", test_values)
print("GELU:", np.round(gelu(test_values), 4))
print("Note: GELU softly suppresses near-zero values instead of hard-zeroing them.")


# --------------------------
# Assignment 3 — Single FFN Forward Pass
# --------------------------
# FFN for one token vector x of shape (d_model,):
#   hidden = activation(x @ W1 + b1)   ← expand to many neurons
#   output = hidden @ W2 + b2           ← compress back to d_model
#
# Each hidden dimension = one neuron = one specialist.

def ffn(x, W1, b1, W2, b2, activation=gelu):
    hidden = activation(x @ W1 + b1)
    output = hidden @ W2 + b2
    return output


print("\n=== Assignment 3: Single FFN Forward Pass ===")

d_model = 8
d_ff = 32  # 4× expansion: more neurons = more specialists

x_single = np.random.randn(d_model)
W1 = np.random.randn(d_model, d_ff)
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model)
b2 = np.zeros(d_model)

output_single = ffn(x_single, W1, b1, W2, b2)
print("Input shape  (d_model)     :", x_single.shape)
print("W1 shape     (d_model→d_ff):", W1.shape, "←", d_ff, "neurons")
print("W2 shape     (d_ff→d_model):", W2.shape)
print("Output shape (d_model)     :", output_single.shape)
print("Output:", np.round(output_single, 4))


# --------------------------
# Assignment 4 — Apply FFN to All Tokens
# --------------------------
# The same FFN is applied independently to each token.
# No information flows between tokens here — that already happened in attention.
# Think of it as: attention gathered context, FFN now examines each token alone.

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
print("Input shape :", x_seq.shape)
print("Output shape:", output_seq.shape)
print("Shapes match:", x_seq.shape == output_seq.shape)

print("\nFirst token:")
print("  Input :", np.round(x_seq[0], 4))
print("  Output:", np.round(output_seq[0], 4))
