import numpy as np

np.random.seed(42)


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


# --------------------------
# Assignment 1 — Low-Rank Matrix Decomposition
# --------------------------

print("=== Assignment 1: Low-Rank Matrix ===")

d = 100
r = 5

W_full = np.random.randn(d, d)

A = np.random.randn(r, d)
B = np.random.randn(d, r)
W_lowrank = B @ A

print(f"Full W:         {d}×{d} = {d*d:,} parameters")
print(f"Low-rank (r={r}): {d}×{r} + {r}×{d} = {d*r + r*d:,} parameters")
print(f"Compression:    {d*d / (2*d*r):.1f}× fewer parameters")

diff = np.linalg.norm(W_full - W_lowrank) / np.linalg.norm(W_full)
print(f"Relative difference from full matrix: {diff:.4f}")
print("(Low-rank is an approximation — the difference is expected)")


# --------------------------
# Assignment 2 — LoRA Layer
# --------------------------

class LoRALayer:
    def __init__(self, W, r=4, alpha=1.0):
        d_out, d_in = W.shape
        self.W = W.copy()          # frozen pretrained weight
        self.A = np.random.randn(r, d_in) * 0.01    # trainable
        self.B = np.zeros((d_out, r))               # trainable, init to 0
        self.scale = alpha / r

    def forward(self, x):
        base = x @ self.W.T
        lora = self.scale * (x @ self.A.T @ self.B.T)
        return base + lora

    def param_count(self):
        d_out, d_in = self.W.shape
        r = self.A.shape[0]
        return r * d_in + d_out * r

    def frozen_param_count(self):
        return self.W.size


print("\n=== Assignment 2: LoRA Layer ===")

d_in, d_out = 64, 64
W_pretrained = np.random.randn(d_out, d_in)

lora = LoRALayer(W_pretrained, r=8, alpha=8.0)

x = np.random.randn(5, d_in)
output = lora.forward(x)
print(f"Input:  {x.shape}")
print(f"Output: {output.shape}")
print(f"Output (first token, first 5 dims): {np.round(output[0, :5], 4)}")

print(f"\nFrozen params (W): {lora.frozen_param_count():,}")
print(f"Trainable params (A+B): {lora.param_count():,}")


# --------------------------
# Assignment 3 — Parameter Count Comparison
# --------------------------

print("\n=== Assignment 3: Parameter Count Comparison ===")

configs = [
    {"d_model": 768, "n_layers": 12, "rank": 8, "name": "GPT-2 Small"},
    {"d_model": 4096, "n_layers": 32, "rank": 16, "name": "LLaMA 2 7B"},
]

for cfg in configs:
    d = cfg["d_model"]
    n = cfg["n_layers"]
    r = cfg["rank"]

    # W_Q, W_K, W_V, W_O per layer
    full_attn_params = n * 4 * d * d
    lora_params = n * 4 * 2 * r * d

    print(f"\n{cfg['name']}:")
    print(f"  Full attention params: {full_attn_params:>15,}")
    print(f"  LoRA params (r={r}):    {lora_params:>15,}")
    print(f"  Reduction:             {full_attn_params / lora_params:.1f}×")


# --------------------------
# Assignment 4 — Train LoRA Adapter
# --------------------------

print("\n=== Assignment 4: Train LoRA Adapter ===")

d_in, d_out = 8, 8
vocab_size = 10
n_samples = 50
learning_rate = 0.01
n_epochs = 200

# Pretrained frozen weights
W_frozen = np.random.randn(d_out, d_in) * 0.5
W_lm = np.random.randn(d_in, vocab_size) * 0.1

# Target behavior: we want the model to prefer certain tokens
X_data = np.random.randn(n_samples, d_in)
y_data = np.random.randint(0, vocab_size, size=n_samples)

lora_train = LoRALayer(W_frozen, r=4, alpha=4.0)


def train_lora_step(lora, X, y, lr):
    # Forward
    h = lora.forward(X)
    logits = h @ W_lm
    probs = softmax(logits)

    # Loss
    batch_size = len(y)
    loss = 0.0
    for i in range(batch_size):
        loss += -np.log(max(probs[i, y[i]], 1e-9))
    loss /= batch_size

    # Gradient w.r.t. logits
    dlogits = probs.copy()
    for i in range(batch_size):
        dlogits[i, y[i]] -= 1
    dlogits /= batch_size

    # Gradient w.r.t. h
    dh = dlogits @ W_lm.T

    # Gradient w.r.t. LoRA params (B and A)
    # h_lora = scale * X @ A.T @ B.T
    # d(h)/d(B) = scale * (X @ A.T)
    # d(h)/d(A) = scale * (B.T @ dh).T @ X ... simplified below

    scale = lora.scale
    X_A = X @ lora.A.T                  # (batch, r)
    dB = scale * (dh.T @ X_A)           # (d_out, r)

    B_dh = dh @ lora.B                  # (batch, r)
    dA = scale * (B_dh.T @ X)           # (r, d_in)

    # Update only A and B (W is frozen)
    lora.B -= lr * dB
    lora.A -= lr * dA

    return loss


print("Training LoRA adapter (W is frozen):")
for epoch in range(n_epochs):
    idx = np.random.choice(n_samples, size=16, replace=False)
    loss = train_lora_step(lora_train, X_data[idx], y_data[idx], learning_rate)
    if epoch % 50 == 0 or epoch == n_epochs - 1:
        print(f"  Epoch {epoch:4d}: loss = {loss:.4f}")

print("\nLoRA adapter trained. W_frozen was never modified.")
print(f"W_frozen changed: {not np.allclose(W_frozen, lora_train.W)}")
