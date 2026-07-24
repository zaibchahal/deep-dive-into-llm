# 002 — Embeddings: Notes

## Key Ideas

- Token IDs are just labels. `72` does not mean more than `15`.
- Embeddings convert each token ID into a dense vector of floats.
- The embedding matrix has shape `(vocab_size, embedding_dim)`.
- Looking up an embedding is just indexing a row: `matrix[token_id]`.
- Embeddings are **learned** during training via backpropagation.
- After training, similar words have similar vectors.

## Cosine Similarity

```
similarity = (A · B) / (|A| × |B|)
```

- Range: -1 to 1
- 1 = identical direction
- 0 = orthogonal (unrelated)
- -1 = opposite

## Real Model Dimensions

| Model   | Embedding Dim |
|---------|---------------|
| GPT-2   | 768           |
| GPT-3   | 12288         |
| Llama 2 | 4096          |

## Why not just use one-hot vectors?

One-hot: `cat = [0,0,1,0,...,0]` — 50,000 dimensions, no meaning.

Embedding: `cat = [0.21, 0.54, -0.13, 0.87]` — 4–12k dimensions, learned meaning.

## What comes next

Embeddings give meaning to tokens but not **position**.

`"dog bites man"` and `"man bites dog"` would have identical embeddings.

That's why we need **Positional Encoding** next.
