import numpy as np
import math

np.random.seed(42)


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores)
    return weights @ V, weights


# --------------------------
# Assignment 1 — Single Head Review
# --------------------------

print("=== Assignment 1: Single Head ===")

seq_len = 4
d_model = 8
h = 4
d_k = d_model // h  # 2 per head

x = np.random.randn(seq_len, d_model)

W_Q0 = np.random.randn(d_model, d_k)
W_K0 = np.random.randn(d_model, d_k)
W_V0 = np.random.randn(d_model, d_k)

Q0 = x @ W_Q0
K0 = x @ W_K0
V0 = x @ W_V0

out0, _ = scaled_dot_product_attention(Q0, K0, V0)
print("Single head output shape:", out0.shape)


# --------------------------
# Assignment 2 — Multiple Heads in Loop
# --------------------------

print("\n=== Assignment 2: Multiple Heads in Loop ===")

W_Qs = [np.random.randn(d_model, d_k) for _ in range(h)]
W_Ks = [np.random.randn(d_model, d_k) for _ in range(h)]
W_Vs = [np.random.randn(d_model, d_k) for _ in range(h)]

head_outputs = []

for i in range(h):
    Q_i = x @ W_Qs[i]
    K_i = x @ W_Ks[i]
    V_i = x @ W_Vs[i]

    out_i, _ = scaled_dot_product_attention(Q_i, K_i, V_i)
    head_outputs.append(out_i)
    print(f"Head {i+1} output shape: {out_i.shape}")


# --------------------------
# Assignment 3 — Concatenate and Project
# --------------------------

print("\n=== Assignment 3: Concatenate and Project ===")

W_O = np.random.randn(d_model, d_model)

concat = np.concatenate(head_outputs, axis=-1)
print("Concat shape:", concat.shape)

output = concat @ W_O
print("Final output shape:", output.shape)


# --------------------------
# Assignment 4 — Full Multi-Head Attention Function
# --------------------------

def multi_head_attention(x, h=4, mask=None):
    seq_len, d_model = x.shape
    d_k = d_model // h

    W_Qs = [np.random.randn(d_model, d_k) for _ in range(h)]
    W_Ks = [np.random.randn(d_model, d_k) for _ in range(h)]
    W_Vs = [np.random.randn(d_model, d_k) for _ in range(h)]
    W_O = np.random.randn(d_model, d_model)

    heads = []
    for i in range(h):
        Q_i = x @ W_Qs[i]
        K_i = x @ W_Ks[i]
        V_i = x @ W_Vs[i]
        head_out, _ = scaled_dot_product_attention(Q_i, K_i, V_i, mask)
        heads.append(head_out)

    concat = np.concatenate(heads, axis=-1)
    return concat @ W_O


print("\n=== Assignment 4: Full Multi-Head Attention ===")
x = np.random.randn(6, 8)
output = multi_head_attention(x, h=4)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Input and output have the same shape:", x.shape == output.shape)
