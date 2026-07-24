# 016 — RAG (Retrieval-Augmented Generation)

## Goal

By the end of this module, you should be able to answer:

* What is RAG and why is it needed?
* How does chunking work and why do we use overlap?
* How are documents embedded and indexed?
* What is the difference between bi-encoder and cross-encoder?
* How does BM25 work and why is hybrid search better than dense-only?
* What is Reciprocal Rank Fusion?
* How does metadata filtering narrow retrieval?
* How is context assembled from reranked results?

---

# Theory

## 1. The Problem: Knowledge Cutoff

LLMs are trained on data up to a cutoff date.

They don't know:

* What happened last week
* Your company's internal documents
* The contents of a specific PDF
* Private customer data

---

## 2. RAG: The Solution

Instead of training on the data, **retrieve it at query time**.

```
User question
     ↓
Hybrid search (BM25 + dense)
     ↓
Metadata filter
     ↓
Cross-encoder reranking
     ↓
Context construction
     ↓
LLM generates answer based on retrieved context
```

---

## 3. Full Pipeline

```
Offline (indexing):
  Documents → Chunk → Embed → Store in vector DB

Online (retrieval):
  Query → Embed → Hybrid search → Filter → Rerank → Context → LLM
```

---

## 4. Chunking

Long documents must be split before embedding.

Sliding-window chunker with overlap:

```
chunk_size = 12 words, overlap = 3 words
chunk[0]:  words  0–11
chunk[1]:  words  9–20   ← 3-word overlap
chunk[2]:  words 18–29
```

Overlap prevents facts at chunk boundaries from being lost.

---

## 5. Embeddings (Bi-Encoder)

Each chunk becomes a unit vector:

```
embed(chunk) → v ∈ ℝ^d,  ‖v‖ = 1
sim(query, doc) = q · d   (dot product = cosine for unit vectors)
```

Bi-encoder: query and document encoded **independently** → fast offline indexing.

---

## 6. Vector Database

Stores (vector, metadata) pairs. Supports:

* Insert
* k-NN search
* Filtered search (pre-filter or post-filter on metadata)

---

## 7. Hybrid Search (BM25 + Dense)

**Dense** retrieval catches semantic similarity but misses exact keyword matches.

**BM25** (lexical) catches keyword matches but misses paraphrases.

**Reciprocal Rank Fusion (RRF)** combines both ranked lists:

```
rrf(doc) = Σ_r  1 / (60 + rank_r(doc))
```

---

## 8. Metadata Filtering

Filter by structured fields (category, year, source) before or after retrieval.

```python
vdb.search(q_vec, filter_fn=lambda m: m["category"] == "ai" and m["year"] >= 2023)
```

---

## 9. Reranking (Cross-Encoder)

Two-stage retrieval:

1. **Stage 1** — bi-encoder retrieves top-50 (fast, approximate)
2. **Stage 2** — cross-encoder reranks to top-5 (accurate, models query-doc interaction)

Cross-encoder feeds `[query, doc]` jointly → captures phrase matches and proximity.

---

## 10. Context Construction

Assemble reranked chunks into a prompt:

* Token budget (max words to include)
* Deduplication (Jaccard similarity ≥ 0.5 → skip redundant chunks)
* Source attribution in prompt

---

# Coding Assignments

## Assignment 1 — Chunking

Implement a sliding-window word-level chunker with configurable `chunk_size` and `overlap`.

Verify that the last `overlap` words of chunk[i] equal the first `overlap` words of chunk[i+1].

---

## Assignment 2 — Embeddings

Create a deterministic pseudo-embedding function that maps text to a unit vector.

Verify that the embedding matrix is shape `(N_docs, dim)` and all rows have norm ≈ 1.

---

## Assignment 3 — Vector Database

Build a `VectorDB` class supporting:
- `add(vector, metadata)`
- `search(query_vec, top_k, filter_fn=None) → [(score, metadata)]`

---

## Assignment 4 — Similarity Search

Implement cosine k-NN retrieval. Show top results for three different queries.

---

## Assignment 5 — Hybrid Search (BM25 + Dense)

Implement BM25 from scratch (IDF, TF, length normalisation).

Implement RRF to fuse BM25 and dense rankings.

Show top-5 results for each method and compare.

---

## Assignment 6 — Metadata Filtering

Demonstrate:
- No filter
- Single field filter (`category == "ai"`)
- Composite filter (`category == "ai"` AND `year >= 2023`)
- Source filter

---

## Assignment 7 — Reranking (Cross-Encoder)

Implement a cross-encoder scoring function (term overlap + position weighting + phrase bonus).

Run Stage 1 retrieval (top-8), then rerank. Show rank changes — reranking should lift
the most relevant document even if it ranked low in Stage 1.

---

## Assignment 8 — Context Construction

Implement context assembly with:
- Token budget
- Jaccard-based deduplication
- Source-tagged prompt template

Run the full end-to-end pipeline for two different queries.

---

# Success Criteria

* Understand why RAG exists
* Implement sliding-window chunking with overlap
* Embed documents and store in a vector DB
* Implement BM25 from scratch
* Fuse rankings with RRF
* Filter by metadata fields
* Rerank with a cross-encoder and understand rank changes
* Build a token-budget-aware context with deduplication
* Know when to use RAG vs fine-tuning
