import numpy as np
import math

np.random.seed(42)

# --------------------------
# Assignment 1 — Prepare Data
# --------------------------

print("=== Assignment 1: Prepare Data ===")

training_text = (
    "the cat sat on the mat. "
    "the cat ate the rat. "
    "the rat ran from the cat. "
    "the mat is flat. "
    "the cat is fat. "
)

# Build character vocabulary
chars = sorted(set(training_text))
vocab_size = len(chars)
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for c, i in char_to_id.items()}

print("Vocabulary:", ''.join(chars))
print("Vocab size:", vocab_size)

# Encode full text
encoded = [char_to_id[c] for c in training_text]
print("Encoded (first 20):", encoded[:20])

# Create training pairs with sliding window
context_len = 8

def make_batches(encoded, context_len):
    X, Y = [], []
    for i in range(len(encoded) - context_len):
        X.append(encoded[i: i + context_len])
        Y.append(encoded[i + 1: i + context_len + 1])
    return np.array(X), np.array(Y)

X_train, Y_train = make_batches(encoded, context_len)
print(f"\nTraining samples: {len(X_train)}")
print("X[0]:", X_train[0], "→", ''.join(id_to_char[i] for i in X_train[0]))
print("Y[0]:", Y_train[0], "→", ''.join(id_to_char[i] for i in Y_train[0]))


# --------------------------
# Assignment 2 — Tiny Model (Embedding + LM Head only)
# (Simplified: skips full transformer for CPU speed)
# --------------------------

print("\n=== Assignment 2: Tiny Model (Bigram + Embedding) ===")

d_model = 32

# Parameters
embedding = np.random.randn(vocab_size, d_model) * 0.01
W_lm = np.random.randn(d_model, vocab_size) * 0.01


def softmax_2d(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def forward(token_ids):
    # (seq_len, d_model) — simple embedding lookup only
    x = embedding[token_ids]
    # (seq_len, vocab_size) — project to vocab
    logits = x @ W_lm
    return logits


def compute_loss(logits, targets):
    probs = softmax_2d(logits)
    seq_len = len(targets)
    loss = 0.0
    for i, t in enumerate(targets):
        p = max(probs[i][t], 1e-9)
        loss += -np.log(p)
    return loss / seq_len


# --------------------------
# Assignment 3 — Training Loop
# --------------------------

print("\n=== Assignment 3: Training Loop ===")

learning_rate = 0.05
n_epochs = 100
sample_size = 32
eps = 1e-4

def train_step(x_batch, y_batch):
    global embedding, W_lm

    # Numerical gradient for embedding rows used in this batch
    logits = forward(x_batch)
    loss = compute_loss(logits, y_batch)

    # Gradient of LM head (W_lm) via closed-form softmax cross-entropy
    probs = softmax_2d(logits)
    dlogits = probs.copy()
    for i, t in enumerate(y_batch):
        dlogits[i][t] -= 1
    dlogits /= len(y_batch)

    # W_lm gradient: x.T @ dlogits  (x = embedding[x_batch])
    x = embedding[x_batch]
    dW_lm = x.T @ dlogits

    # Embedding gradient: dlogits @ W_lm.T
    dx = dlogits @ W_lm.T

    # Update W_lm
    W_lm -= learning_rate * dW_lm

    # Update only embedding rows used
    for i, token_id in enumerate(x_batch):
        embedding[token_id] -= learning_rate * dx[i]

    return loss


loss_history = []

for epoch in range(n_epochs):
    # Random mini-batch
    idx = np.random.choice(len(X_train), size=min(sample_size, len(X_train)), replace=False)
    x_batch = X_train[idx[0]]
    y_batch = Y_train[idx[0]]

    loss = train_step(x_batch, y_batch)
    loss_history.append(loss)

    if epoch % 20 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch:4d}: loss = {loss:.4f}")

print(f"\nLoss reduced from {loss_history[0]:.4f} to {loss_history[-1]:.4f}")


# --------------------------
# Assignment 4 — Generate Text
# --------------------------

def generate(start_char, max_chars=40):
    tokens = [char_to_id.get(start_char, 0)]
    result = [start_char]

    for _ in range(max_chars):
        logits = forward(tokens[-context_len:] if len(tokens) >= context_len else tokens)
        last_logits = logits[-1]
        probs = softmax_2d(last_logits.reshape(1, -1))[0]
        next_token = int(np.random.choice(vocab_size, p=probs))
        tokens.append(next_token)
        result.append(id_to_char[next_token])

    return ''.join(result)


print("\n=== Assignment 4: Generate Text ===")
print("Generating text after training:")
for start in ['t', 'c', 'm']:
    print(f"  Starting with '{start}': {generate(start, 50)}")
