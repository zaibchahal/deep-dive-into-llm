import numpy as np
import math

np.random.seed(42)


# ---- Helper functions from previous modules ----

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def causal_mask(seq_len):
    return np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)


def multi_head_attention(x, W_Qs, W_Ks, W_Vs, W_O, mask=None):
    h = len(W_Qs)
    d_k = W_Qs[0].shape[1]
    heads = []
    for i in range(h):
        Q_i = x @ W_Qs[i]
        K_i = x @ W_Ks[i]
        V_i = x @ W_Vs[i]
        scores = Q_i @ K_i.T / math.sqrt(d_k)
        if mask is not None:
            scores = scores + mask
        weights = softmax(scores)
        heads.append(weights @ V_i)
    concat = np.concatenate(heads, axis=-1)
    return concat @ W_O


def ffn(x, W1, b1, W2, b2):
    return gelu(x @ W1 + b1) @ W2 + b2


# ---- Transformer Block Parameters ----

class TransformerBlockParams:
    def __init__(self, d_model, d_ff, h):
        d_k = d_model // h
        self.W_Qs = [np.random.randn(d_model, d_k) * 0.02 for _ in range(h)]
        self.W_Ks = [np.random.randn(d_model, d_k) * 0.02 for _ in range(h)]
        self.W_Vs = [np.random.randn(d_model, d_k) * 0.02 for _ in range(h)]
        self.W_O  = np.random.randn(d_model, d_model) * 0.02
        self.W1   = np.random.randn(d_model, d_ff) * 0.02
        self.b1   = np.zeros(d_ff)
        self.W2   = np.random.randn(d_ff, d_model) * 0.02
        self.b2   = np.zeros(d_model)
        self.gamma1 = np.ones(d_model)
        self.beta1  = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2  = np.zeros(d_model)


# --------------------------
# Assignment 1 — Full Transformer Block
# --------------------------

def transformer_block(x, params, use_causal_mask=True, verbose=False):
    seq_len = x.shape[0]

    mask = causal_mask(seq_len) if use_causal_mask else None

    # Sublayer 1: LayerNorm → Attention → Residual
    x_norm = layer_norm(x, params.gamma1, params.beta1)
    if verbose:
        print("  After LayerNorm 1:", x_norm.shape)

    attn_out = multi_head_attention(x_norm, params.W_Qs, params.W_Ks, params.W_Vs, params.W_O, mask)
    if verbose:
        print("  After Attention:", attn_out.shape)

    x = x + attn_out
    if verbose:
        print("  After Residual 1:", x.shape)

    # Sublayer 2: LayerNorm → FFN → Residual
    x_norm = layer_norm(x, params.gamma2, params.beta2)
    if verbose:
        print("  After LayerNorm 2:", x_norm.shape)

    ffn_out = ffn(x_norm, params.W1, params.b1, params.W2, params.b2)
    if verbose:
        print("  After FFN:", ffn_out.shape)

    x = x + ffn_out
    if verbose:
        print("  After Residual 2:", x.shape)

    return x


# --------------------------
# Assignment 2 — Shape Verification
# --------------------------

print("=== Assignment 1 & 2: Transformer Block + Shape Verification ===")

d_model = 16
d_ff = 64
h = 4

params = TransformerBlockParams(d_model, d_ff, h)

for seq_len in [1, 4, 10]:
    x = np.random.randn(seq_len, d_model)
    out = transformer_block(x, params)
    assert x.shape == out.shape
    print(f"seq_len={seq_len}: input={x.shape} → output={out.shape} ✓")


# --------------------------
# Assignment 3 — Print Intermediate Shapes
# --------------------------

print("\n=== Assignment 3: Intermediate Shapes ===")

x = np.random.randn(5, d_model)
print(f"Input: {x.shape}")
out = transformer_block(x, params, verbose=True)
print(f"Output: {out.shape}")
