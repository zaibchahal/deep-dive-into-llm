import numpy as np
import time

np.random.seed(42)


def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


vocab_size = 20
d_model = 16
context_len = 8

# Toy model weights
embedding = np.random.randn(vocab_size, d_model) * 0.1
W_lm = np.random.randn(d_model, vocab_size) * 0.1
W_Q = np.random.randn(d_model, d_model) * 0.1
W_K = np.random.randn(d_model, d_model) * 0.1
W_V = np.random.randn(d_model, d_model) * 0.1


def embed_tokens(token_ids):
    return embedding[token_ids]


def compute_kv(x):
    K = x @ W_K
    V = x @ W_V
    return K, V


def compute_q(x):
    return x @ W_Q


def attention_with_kv(Q, K, V):
    scores = Q @ K.T / np.sqrt(d_model)
    scores = scores - scores.max()
    weights = np.exp(scores) / np.sum(np.exp(scores))
    return weights @ V


def lm_head(x):
    return x @ W_lm


# --------------------------
# Assignment 1 — Simple Inference Function
# --------------------------

def inference(prompt_ids, max_new_tokens=20):
    tokens = list(prompt_ids)
    for _ in range(max_new_tokens):
        window = tokens[-context_len:]
        x = embed_tokens(window)
        logits = lm_head(x[-1])
        probs = softmax(logits)
        next_token = int(np.random.choice(vocab_size, p=probs))
        tokens.append(next_token)
        if next_token == 0:  # treat 0 as <EOS>
            break
    return tokens


print("=== Assignment 1: Simple Inference ===")
prompt = [3, 7, 2]
generated = inference(prompt, max_new_tokens=10)
print("Prompt:", prompt)
print("Generated:", generated)
print("New tokens:", generated[len(prompt):])


# --------------------------
# Assignment 2 — KV Cache Simulation
# --------------------------

def inference_without_cache(prompt_ids, n_steps):
    tokens = list(prompt_ids)
    for _ in range(n_steps):
        x = embed_tokens(tokens[-context_len:])
        K_full = x @ W_K
        V_full = x @ W_V
        Q_last = (x[-1:]) @ W_Q
        out = attention_with_kv(Q_last, K_full, V_full)
        logits = lm_head(out[0])
        probs = softmax(logits)
        next_token = int(np.argmax(probs))
        tokens.append(next_token)
    return tokens


def inference_with_cache(prompt_ids, n_steps):
    tokens = list(prompt_ids)

    # Prefill: compute K, V for all prompt tokens
    x_prompt = embed_tokens(tokens)
    K_cache = x_prompt @ W_K
    V_cache = x_prompt @ W_V

    for _ in range(n_steps):
        x_new = embed_tokens([tokens[-1]])
        K_new = x_new @ W_K
        V_new = x_new @ W_V

        K_cache = np.vstack([K_cache, K_new])
        V_cache = np.vstack([V_cache, V_new])

        Q_last = x_new @ W_Q
        out = attention_with_kv(Q_last, K_cache, V_cache)
        logits = lm_head(out[0])
        probs = softmax(logits)
        next_token = int(np.argmax(probs))
        tokens.append(next_token)

    return tokens


print("\n=== Assignment 2: KV Cache Timing ===")

prompt = [3, 7, 2, 5, 1]
n_steps = 30

start = time.time()
for _ in range(50):
    inference_without_cache(prompt, n_steps)
t_no_cache = time.time() - start

start = time.time()
for _ in range(50):
    inference_with_cache(prompt, n_steps)
t_with_cache = time.time() - start

print(f"Without KV cache: {t_no_cache:.4f}s")
print(f"With KV cache:    {t_with_cache:.4f}s")
print(f"Speedup: {t_no_cache / t_with_cache:.2f}x")


# --------------------------
# Assignment 3 — Batch Inference
# --------------------------

def batch_inference(prompts, max_new_tokens=10, pad_id=0):
    max_len = max(len(p) for p in prompts)
    padded = []
    for p in prompts:
        padded.append([pad_id] * (max_len - len(p)) + p)
    padded = np.array(padded)

    outputs = []
    for row in padded:
        x = embed_tokens(row)
        logits = lm_head(x[-1])
        probs = softmax(logits)
        next_token = int(np.argmax(probs))
        outputs.append(list(row) + [next_token])

    return outputs


print("\n=== Assignment 3: Batch Inference ===")

prompts = [
    [3, 7],
    [5, 1, 2, 8],
    [9],
]

results = batch_inference(prompts)
for i, r in enumerate(results):
    print(f"Prompt {i}: {prompts[i]} → next token: {r[-1]}")


# --------------------------
# Assignment 4 — Stop Sequences
# --------------------------

def inference_with_stop(prompt_ids, stop_token=0, max_new_tokens=30):
    tokens = list(prompt_ids)
    for _ in range(max_new_tokens):
        window = tokens[-context_len:]
        x = embed_tokens(window)
        logits = lm_head(x[-1])
        probs = softmax(logits)
        next_token = int(np.random.choice(vocab_size, p=probs))
        tokens.append(next_token)
        if next_token == stop_token:
            print(f"  Stopped at EOS token (id={stop_token})")
            break
    return tokens


print("\n=== Assignment 4: Stop Sequences ===")
result = inference_with_stop([3, 7, 2], stop_token=5, max_new_tokens=20)
print("Generated:", result)
print("Length:", len(result))
