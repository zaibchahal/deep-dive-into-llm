# 014 — Inference: Notes

## Inference vs Training

- Training: compute gradients, update weights
- Inference: frozen weights, generate output only

No backward pass needed. Much faster per forward.

## Two Phases

1. **Prefill**: process prompt tokens in parallel (fast)
2. **Decode**: generate tokens one at a time (slower)

## KV Cache

Without cache: recompute K, V for all tokens at every step → O(n²) work.

With cache: store K, V from previous steps → O(n) work per new token.

Trade-off: memory grows with sequence length.

## Memory Usage

KV cache size:
```
2 × n_layers × seq_len × d_model × bytes_per_float
```

For LLaMA 2 7B at seq_len=4096: ~2GB of KV cache.

## Batched Inference

Process multiple prompts at once.

Pros: better GPU utilization.
Cons: need padding, different sequence lengths.

## Key Inference Parameters

- `max_new_tokens`: max generated tokens
- `temperature`: randomness
- `top_p`, `top_k`: sampling strategy
- `stop_sequences`: custom stop strings

## Latency vs Throughput

- Low latency: small batch, fast first token (chat)
- High throughput: large batch, offline processing (batch jobs)

## Next

**LoRA** — fine-tune a model efficiently without updating all parameters.
