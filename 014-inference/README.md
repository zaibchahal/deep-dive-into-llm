# 014 — Inference

## References

* [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) — Runs a live GPT-2 model in the browser. Type any text and watch the full inference pipeline: tokenization → embeddings → attention → FFN → softmax sampling, all in real time.

---

## Goal

By the end of this module, you should be able to answer:

* What happens during inference vs training?
* What is the KV cache and why does it matter?
* How does prompt processing differ from generation?
* What is batch inference?
* What is latency vs throughput?

---

# Theory

## 1. Inference vs Training

| | Training | Inference |
|-|----------|-----------|
| Goal | Update weights | Generate output |
| Gradients | Required | Not needed |
| Data | Many examples | Single prompt |
| Speed | Slow (OK) | Fast (required) |

During inference:

* Weights are frozen
* No `loss.backward()`
* No optimizer

---

## 2. The Two Phases

### Phase 1: Prefill (Prompt Processing)

Process all prompt tokens in parallel.

Fast — full parallelism.

```
"The cat sat" → [token0, token1, token2] → process all at once
```

### Phase 2: Decode (Generation)

Generate one token at a time.

Slow — must be sequential.

```
→ token3, token4, token5, ...
```

---

## 3. KV Cache

Problem: during generation, we recompute K and V for all previous tokens at every step.

This is wasteful.

Solution: **Cache K and V** from previous steps.

At step t:

```
New token → compute Q_t, K_t, V_t

K_cache = [K_0, K_1, ..., K_{t-1}, K_t]
V_cache = [V_0, V_1, ..., V_{t-1}, V_t]

Attention: Q_t × K_cache.T → attend to all past tokens
```

Only compute attention for the new token.

Memory: KV cache grows with sequence length.

---

## 4. Batched Inference

Process multiple prompts simultaneously.

Increases GPU utilization.

```
Batch = [prompt_1, prompt_2, prompt_3]
→ Forward pass on all three at once
```

Problem: prompts have different lengths (need padding).

---

## 5. Latency vs Throughput

| | Latency | Throughput |
|-|---------|------------|
| Meaning | Time per request | Requests per second |
| Good for | Interactive chat | Batch processing |
| Batch size | Small (1) | Large (32, 128) |

---

## 6. Sampling Parameters at Inference

* `temperature`: creativity vs determinism
* `max_new_tokens`: how many tokens to generate
* `top_p`, `top_k`: restrict sampling distribution
* `stop_sequences`: stop when these strings appear

---

# Coding Assignments

## Assignment 1 — Simple Inference Function

```python
def inference(model, prompt_ids, max_new_tokens=50):
    pass
```

Run forward pass and generate tokens.

---

## Assignment 2 — KV Cache Simulation

Show the time difference between:

* Without KV cache: recompute all K, V at every step
* With KV cache: only compute for new token

---

## Assignment 3 — Batch Inference

Run inference on multiple prompts at once.

Use padding for different lengths.

---

## Assignment 4 — Stop Sequences

Stop generation when a specific token sequence appears.

---

# Success Criteria

* Know inference vs training differences
* Understand KV cache and why it speeds up generation
* Implement simple inference with caching
* Know latency vs throughput trade-off
