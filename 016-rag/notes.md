# 016 — RAG: Notes

## What RAG Solves

LLMs are static after training. They don't know:
- Events after their cutoff
- Your private documents
- Domain-specific or proprietary data

RAG = retrieve external knowledge at query time, inject into the prompt.

```
Offline:  docs → chunk → embed → vector store
Online:   query → embed → hybrid search → filter → rerank → context → LLM
```

---

## 1. Chunking

Long documents must be split before embedding.

```
sliding window:
  chunk_size = 12 words
  overlap    = 3 words

[0]  word_0 … word_11
[1]  word_9 … word_20   ← 3-word overlap keeps context across boundaries
[2]  word_18 … word_29
```

Why overlap? A fact at the boundary of two chunks would be lost without it.

Typical production values: 256–512 tokens, 10–20% overlap.

---

## 2. Embeddings (Bi-Encoder)

Each chunk becomes a unit vector.

```
embed("Eiffel Tower is 330m tall") → v_doc    ∈ ℝ^d, ‖v_doc‖ = 1
embed("how tall is the Eiffel Tower?") → v_query ∈ ℝ^d, ‖v_query‖ = 1

sim = v_query · v_doc   (cosine, since unit norm)
```

Bi-encoder: query and doc are encoded **separately**.
Fast at retrieval time (pre-compute doc vectors offline).

Production: `text-embedding-3-small` (OpenAI), `all-MiniLM-L6-v2` (HuggingFace).

---

## 3. Vector Database

Stores (vector, metadata) pairs. Supports:
- **Insert**: add (vector, payload)
- **k-NN search**: return k vectors closest to query
- **Filtered search**: apply metadata predicate before / after ANN

```
vdb.add(vector, {"id": 5, "category": "ai", "year": 2023, "text": "..."})
vdb.search(q_vec, top_k=5, filter_fn=lambda m: m["year"] >= 2023)
```

Production: Pinecone, Weaviate, Qdrant, pgvector, Chroma.

For very large corpora, exact k-NN is too slow.
Use **HNSW** (Hierarchical Navigable Small World) for approximate nearest-neighbour in O(log N).

---

## 4. Similarity Search

Cosine similarity between unit vectors = dot product.

```
sim(q, d) = q · d / (‖q‖ · ‖d‖)
```

For k-NN retrieval: score all N docs, sort, return top-k.
Production: FAISS, ScaNN, HNSW for sub-linear search.

---

## 5. Hybrid Search (BM25 + Dense)

Dense retrieval (semantic) misses exact keyword matches.
BM25 (lexical) misses paraphrase / synonym matches.
Hybrid combines both.

### BM25

Probabilistic ranking function from information retrieval.

```
score(q, d) = Σ_{t ∈ q}  IDF(t) · tf(t,d)·(k1+1) / [tf(t,d) + k1·(1 − b + b·|d|/avgdl)]

IDF(t) = log( (N − df(t) + 0.5) / (df(t) + 0.5) + 1 )

k1 = 1.5   (TF saturation — diminishing returns for repeated terms)
b  = 0.75  (length normalisation)
```

### Reciprocal Rank Fusion (RRF)

Fuses multiple ranked lists without needing normalised scores.

```
rrf(d) = Σ_r  1 / (k + rank_r(d))    k = 60 (typical)
```

Higher RRF = higher combined rank across both lists.

---

## 6. Metadata Filtering

Documents carry structured metadata alongside their vector:
- `category`, `source`, `year`, `author`, `language`, …

**Pre-filter**: restrict search to a subset before ANN.
```python
vdb.search(q_vec, filter_fn=lambda m: m["category"] == "ai" and m["year"] >= 2023)
```

**Post-filter**: retrieve broadly, then discard non-matching results.

Pre-filter is more efficient; post-filter is simpler to implement.

---

## 7. Reranking (Cross-Encoder)

Two-stage retrieval:

```
Stage 1 — Bi-encoder (fast, approximate):   top-50 candidates
Stage 2 — Cross-encoder (accurate, slow):   top-5 final
```

### Why Two Stages?

Bi-encoder encodes query and doc **independently** → can't model their interaction.

Cross-encoder feeds `[CLS] query [SEP] doc [SEP]` to BERT jointly → captures exact term overlap, proximity, phrase matches.

Cross-encoder is ~100× slower, so run it only on the shortlist.

### Scoring Components (simulated)

1. **Term overlap**: fraction of query terms found in the doc
2. **Position-weighted overlap**: matches near the start score higher
3. **Bigram bonus**: consecutive query terms found together (phrase match)

Real models: `cross-encoder/ms-marco-MiniLM-L-6-v2`, `bge-reranker-large`.

---

## 8. Context Construction

Assemble the reranked chunks into a prompt-ready string.

```
Constraints:
  token_budget: max words to include (e.g. 150)
  dedup:        drop chunks with Jaccard overlap ≥ 0.5 (near-duplicates)

Algorithm:
  for each reranked chunk (best first):
    if budget allows: include it
    elif partial fits: truncate and include
    else: stop
```

### Jaccard Similarity (deduplication)

```
J(A, B) = |A ∩ B| / |A ∪ B|   (token sets)
```

Threshold ≥ 0.5 → chunks share more than half their vocabulary → one is redundant.

### Prompt Template

```
Answer the question based only on the context below.
If the context does not contain the answer, say "I don't know".

Context:
[1] [wikipedia] The Eiffel Tower is 330 metres tall, located in Paris.
[2] [paper] ...

Question: How tall is the Eiffel Tower?

Answer:
```

"Based only on context" reduces hallucination.
Source tags enable citation.

---

## Full Pipeline

```
Documents
    ↓ Chunk           sliding window, overlap
    ↓ Embed           bi-encoder → unit vectors
    ↓ Index           VectorDB (HNSW in production)
                                         Query
                                           ↓ Embed
                                           ↓ Hybrid search  BM25 + dense → RRF
                                           ↓ Metadata filter
                                           ↓ Rerank         cross-encoder
                                           ↓ Context build  dedup + budget
                                           ↓ Prompt
                                           ↓ LLM → Answer
```

## RAG vs Fine-Tuning

| | RAG | Fine-Tuning |
|---|---|---|
| Update cost | None (add to DB) | Full training run |
| Freshness | Always up-to-date | Static after cutoff |
| Fact accuracy | High (exact text) | Can hallucinate |
| Behaviour change | No | Yes |
| Private data | Easy | Risk of leakage |

Use RAG for: facts, docs, citations, private data.
Use fine-tuning for: style, tone, task format, new capabilities.

## Next

**Tool Calling** — let the LLM invoke external functions (search, code execution, APIs) to gather information dynamically rather than from a pre-built index.
