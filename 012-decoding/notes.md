# 012 — Decoding: Notes

## Generation Loop

```
prompt → model → logits → sample next token → append → repeat
```

Autoregressive: each new token is conditioned on all previous tokens.

## Strategies

### Greedy
```
next = argmax(probs)
```
Deterministic, repetitive.

### Temperature
```
probs = softmax(logits / T)
```
T < 1: focused, T > 1: creative.

### Top-K
Keep top K tokens, zero out rest, renormalize, sample.

K=50 is common.

### Top-P (Nucleus)
Keep smallest set with cumulative prob ≥ p, sample.

p=0.9 is common. Adapts K dynamically.

## When to Use

| Use Case | Strategy |
|----------|----------|
| Factual answers | Low temp or greedy |
| Creative writing | High temp + top-p |
| Code generation | Low temp, top-k |
| Chatbots | top-p + temp=0.8 |

## Key Numbers

- GPT-3 API defaults: temperature=1.0, top_p=1.0
- ChatGPT: typically temperature≈0.8

## Stop Tokens

Generation stops at `<EOS>` or max length.

## Next

**Train Tiny GPT** — put all the pieces together and train a small GPT from scratch.
