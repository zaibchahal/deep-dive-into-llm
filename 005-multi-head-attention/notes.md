# 005 — Multi-Head Attention: Notes

## Why Multiple Heads?

One attention head can only capture one type of relationship.

Multiple heads capture different patterns simultaneously:
- Head 1: syntactic relationships
- Head 2: semantic similarity
- Head 3: coreference (what "it" refers to)
- etc.

## Dimension Split

```
d_model = 512, h = 8 → d_k = 64 per head
```

Total params ≈ same as one big head, but richer.

## Formula

```
MultiHead(Q,K,V) = Concat(head_1,...,head_h) @ W_O

head_i = Attention(Q@W_Qi, K@W_Ki, V@W_Vi)
```

## Shape at Each Step

```
Input:           (seq, d_model)
Per head Q,K,V:  (seq, d_k)      where d_k = d_model/h
Head output:     (seq, d_k)
Concat:          (seq, d_model)   ← all heads joined
After W_O:       (seq, d_model)   ← final output
```

## Key Facts

- Output shape = Input shape
- Each head has its own W_Q, W_K, W_V
- W_O projects concatenated heads back to d_model
- In practice, done efficiently with batched matrix ops

## Real Model Example

GPT-3:
- d_model = 12288
- h = 96
- d_k = 128 per head

## Next

**Feed-Forward Network** — after attention, each token is processed independently through a small network.
