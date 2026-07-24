# 016 — RAG: Notes

## Problem RAG Solves

LLMs have a knowledge cutoff.

They don't know your private documents.

RAG = retrieve external knowledge at query time, inject into prompt.

## Pipeline

```
Offline:
  docs → chunk → embed → vector store

Online:
  query → embed → similarity search → top-K docs → prompt → LLM
```

## Embedding for Retrieval

Each chunk becomes a vector.

Query becomes a vector.

Nearest vectors = most relevant chunks.

## Cosine Similarity

```
sim(q, d) = (q · d) / (|q| × |d|)
```

Range -1 to 1. Higher = more similar.

## Chunking Strategy

```
chunk_size = 512 tokens
overlap = 50 tokens
```

Overlap prevents information loss at boundaries.

## Prompt Template

```
Answer using only the context below:

Context: {retrieved_docs}

Question: {question}

Answer:
```

"Only use context" reduces hallucination.

## RAG vs Fine-Tuning

| RAG | Fine-Tuning |
|-----|-------------|
| No retraining | Expensive retraining |
| Up-to-date | Static after training |
| Cites sources | May hallucinate |
| Behavior unchanged | Behavior changes |

## Real RAG Stack

- Embeddings: `text-embedding-3-small` (OpenAI), `all-MiniLM` (HuggingFace)
- Vector DB: Pinecone, Weaviate, pgvector, Chroma
- LLM: GPT-4, Claude, LLaMA

## Next

**Tool Calling** — let the LLM call external functions to fetch data, run code, etc.
