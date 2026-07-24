import numpy as np

np.random.seed(42)


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


# --------------------------
# Assignment 1 — LM Head
# --------------------------

print("=== Assignment 1: LM Head ===")

d_model = 16
vocab_size = 50
seq_len = 5

context_vectors = np.random.randn(seq_len, d_model)

W_lm = np.random.randn(d_model, vocab_size) * 0.02

logits = context_vectors @ W_lm
probs = softmax(logits)

print("Context vectors shape:", context_vectors.shape)
print("W_lm shape:", W_lm.shape)
print("Logits shape:", logits.shape)
print("Probs shape:", probs.shape)
print("Probs row 0 sums to:", round(probs[0].sum(), 6))


# --------------------------
# Assignment 2 — Top Prediction
# --------------------------

print("\n=== Assignment 2: Top Prediction ===")

next_token_id = np.argmax(probs[-1])
next_token_prob = probs[-1][next_token_id]

print("Last token's top prediction:")
print(f"  Token ID: {next_token_id}")
print(f"  Probability: {next_token_prob:.4f}")

top5 = np.argsort(probs[-1])[::-1][:5]
print("\nTop 5 predictions:")
for rank, tid in enumerate(top5):
    print(f"  {rank+1}. Token {tid}: {probs[-1][tid]:.4f}")


# --------------------------
# Assignment 3 — Cross-Entropy Loss
# --------------------------

def cross_entropy(logits, targets):
    probs = softmax(logits)
    losses = []
    for i, target in enumerate(targets):
        correct_prob = probs[i][target]
        correct_prob = max(correct_prob, 1e-9)
        losses.append(-np.log(correct_prob))
    return np.mean(losses)


print("\n=== Assignment 3: Cross-Entropy Loss ===")

# Simulate: predict next token for positions 0..3 (targets are positions 1..4)
token_ids = [3, 7, 2, 1, 15]
input_ids = token_ids[:-1]   # [3, 7, 2, 1]
target_ids = token_ids[1:]   # [7, 2, 1, 15]

seq_context = np.random.randn(len(input_ids), d_model)
logits_for_loss = seq_context @ W_lm

loss = cross_entropy(logits_for_loss, target_ids)
print(f"Target IDs: {target_ids}")
print(f"Cross-entropy loss: {loss:.4f}")
print(f"Random baseline (log(vocab_size)): {np.log(vocab_size):.4f}")


# --------------------------
# Assignment 4 — Loss on a Batch (multiple sequences)
# --------------------------

print("\n=== Assignment 4: Batch Loss ===")

def batch_loss(sequences, W_lm, d_model):
    total_loss = 0
    for seq in sequences:
        if len(seq) < 2:
            continue
        x = np.random.randn(len(seq) - 1, d_model)
        logits = x @ W_lm
        targets = seq[1:]
        total_loss += cross_entropy(logits, targets)
    return total_loss / len(sequences)


sequences = [
    [3, 7, 2, 1, 15],
    [5, 8, 11, 4],
    [0, 3, 2, 9, 7, 12],
]

avg_loss = batch_loss(sequences, W_lm, d_model)
print(f"Average loss over {len(sequences)} sequences: {avg_loss:.4f}")
print("This measures how well the model predicts next tokens on average.")
