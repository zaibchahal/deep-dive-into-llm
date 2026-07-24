import numpy as np

np.random.seed(42)


def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


# --------------------------
# Assignment 1 — Greedy Decoding
# --------------------------

def greedy_decode(logits):
    return int(np.argmax(logits))


print("=== Assignment 1: Greedy Decoding ===")
logits = np.array([0.1, 0.5, 2.3, 0.8, -0.2])
token = greedy_decode(logits)
print("Logits:", logits)
print("Greedy next token:", token, "(always picks token 2 with highest logit)")


# --------------------------
# Assignment 2 — Temperature Sampling
# --------------------------

def temperature_sample(logits, temperature=1.0):
    scaled = logits / temperature
    probs = softmax(scaled)
    return int(np.random.choice(len(probs), p=probs))


print("\n=== Assignment 2: Temperature Sampling ===")
logits = np.array([1.0, 2.0, 3.0, 1.5, 0.5])

print("Temperature=0.1 (almost greedy):")
counts = {}
for _ in range(1000):
    t = temperature_sample(logits, temperature=0.1)
    counts[t] = counts.get(t, 0) + 1
print("  Token frequencies:", dict(sorted(counts.items())))

print("Temperature=2.0 (more random):")
counts = {}
for _ in range(1000):
    t = temperature_sample(logits, temperature=2.0)
    counts[t] = counts.get(t, 0) + 1
print("  Token frequencies:", dict(sorted(counts.items())))


# --------------------------
# Assignment 3 — Top-K Sampling
# --------------------------

def top_k_sample(logits, k=5):
    top_k_indices = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_indices]
    probs = softmax(top_k_logits)
    chosen = np.random.choice(top_k_indices, p=probs)
    return int(chosen)


print("\n=== Assignment 3: Top-K Sampling (k=3) ===")
logits = np.array([0.1, 0.5, 2.3, 0.8, -0.2, 1.1, 0.3])
counts = {}
for _ in range(1000):
    t = top_k_sample(logits, k=3)
    counts[t] = counts.get(t, 0) + 1
print("Logits:", logits)
print("Only top-3 tokens sampled:", dict(sorted(counts.items())))


# --------------------------
# Assignment 4 — Top-P (Nucleus) Sampling
# --------------------------

def top_p_sample(logits, p=0.9):
    probs = softmax(logits)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    cumulative = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative, p) + 1

    nucleus_indices = sorted_indices[:cutoff]
    nucleus_probs = probs[nucleus_indices]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()

    chosen = np.random.choice(nucleus_indices, p=nucleus_probs)
    return int(chosen)


print("\n=== Assignment 4: Top-P Sampling (p=0.9) ===")
logits = np.array([0.1, 0.5, 2.3, 0.8, -0.2, 1.1, 0.3])
counts = {}
for _ in range(1000):
    t = top_p_sample(logits, p=0.9)
    counts[t] = counts.get(t, 0) + 1
print("Logits:", logits)
print("Nucleus sampling distribution:", dict(sorted(counts.items())))


# --------------------------
# Assignment 5 — Generation Loop
# --------------------------

vocab = {0: "<PAD>", 1: "the", 2: "cat", 3: "sat", 4: "on", 5: "mat", 6: "<EOS>"}
vocab_size = len(vocab)


def dummy_model(token_ids):
    np.random.seed(sum(token_ids) % 100)
    return np.random.randn(vocab_size)


def generate(prompt_ids, model_fn, max_new_tokens=10, strategy="greedy", temperature=1.0, k=5, p=0.9):
    tokens = list(prompt_ids)
    for _ in range(max_new_tokens):
        logits = model_fn(tokens)

        if strategy == "greedy":
            next_token = greedy_decode(logits)
        elif strategy == "temperature":
            next_token = temperature_sample(logits, temperature)
        elif strategy == "top_k":
            next_token = top_k_sample(logits, k)
        elif strategy == "top_p":
            next_token = top_p_sample(logits, p)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        tokens.append(next_token)

        if next_token == 6:  # <EOS>
            break

    return tokens


print("\n=== Assignment 5: Generation Loop ===")
prompt = [1, 2]  # "the cat"

for strategy in ["greedy", "temperature", "top_k", "top_p"]:
    generated = generate(prompt, dummy_model, max_new_tokens=8, strategy=strategy)
    words = [vocab.get(t, f"[{t}]") for t in generated]
    print(f"{strategy:12s}: {' '.join(words)}")
