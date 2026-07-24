import numpy as np
import math

np.random.seed(42)

# --------------------------
# Assignment 1 — Q, K, V Projections
# --------------------------

print("=== Assignment 1: Q, K, V Projections ===")

seq_len = 4
d_model = 8
d_k = 8

x = np.random.randn(seq_len, d_model)

W_Q = np.random.randn(d_model, d_k)
W_K = np.random.randn(d_model, d_k)
W_V = np.random.randn(d_model, d_k)

Q = x @ W_Q
K = x @ W_K
V = x @ W_V

print("Input x shape:", x.shape)
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)


# --------------------------
# Assignment 2 — Attention Scores
# --------------------------

print("\n=== Assignment 2: Attention Scores ===")

scores = Q @ K.T
print("Scores shape:", scores.shape)
print("Scores:\n", np.round(scores, 2))


# --------------------------
# Assignment 3 — Softmax
# --------------------------

def softmax(x):
    # Subtract max for numerical stability
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


print("\n=== Assignment 3: Softmax ===")
weights = softmax(scores)
print("Attention weights (each row sums to 1):")
print(np.round(weights, 4))
print("Row sums:", np.round(weights.sum(axis=1), 4))


# --------------------------
# Assignment 4 — Scaled Dot-Product Attention
# --------------------------

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores)
    output = weights @ V
    return output, weights


print("\n=== Assignment 4: Scaled Dot-Product Attention ===")
output, attn_weights = attention(Q, K, V)
print("Output shape:", output.shape)
print("Attention weights:\n", np.round(attn_weights, 4))


# --------------------------
# Assignment 5 — Causal Masking
# --------------------------

def causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = mask * (-1e9)
    return mask


print("\n=== Assignment 5: Causal Masking ===")
mask = causal_mask(seq_len)
print("Causal mask:\n", mask)

output_masked, attn_masked = attention(Q, K, V, mask=mask)
print("\nAttention weights with causal mask:")
print(np.round(attn_masked, 4))
print("Token 0 only sees itself. Token 3 sees all. Correct!")
