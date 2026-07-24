import math
import numpy as np

np.random.seed(42)


# ==============================================================================
# Corpus — documents with metadata
# ==============================================================================

CORPUS = [
    {
        "id": 0,
        "text": "Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability and uses significant whitespace.",
        "source": "wikipedia",
        "category": "programming",
        "year": 2023,
    },
    {
        "id": 1,
        "text": "NumPy is a Python library for numerical computing. It provides support for large multi-dimensional arrays and matrices, along with mathematical functions.",
        "source": "docs",
        "category": "programming",
        "year": 2023,
    },
    {
        "id": 2,
        "text": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. It is 330 metres tall and was built in 1889.",
        "source": "wikipedia",
        "category": "history",
        "year": 2022,
    },
    {
        "id": 3,
        "text": "The Great Wall of China stretches over 21,000 kilometres. Construction began in the 7th century BC and continued for many centuries.",
        "source": "wikipedia",
        "category": "history",
        "year": 2022,
    },
    {
        "id": 4,
        "text": "GPT-4 is a large multimodal model released by OpenAI in March 2023. It accepts image and text inputs and produces text outputs.",
        "source": "openai",
        "category": "ai",
        "year": 2023,
    },
    {
        "id": 5,
        "text": "Transformer models use self-attention mechanisms to process sequences. They replaced recurrent networks in most NLP tasks.",
        "source": "paper",
        "category": "ai",
        "year": 2022,
    },
    {
        "id": 6,
        "text": "Machine learning is a subset of artificial intelligence. Models learn patterns from data without being explicitly programmed.",
        "source": "textbook",
        "category": "ai",
        "year": 2021,
    },
    {
        "id": 7,
        "text": "Paris is the capital and most populous city of France. It sits along the Seine river and has a population of over 2 million.",
        "source": "wikipedia",
        "category": "geography",
        "year": 2023,
    },
    {
        "id": 8,
        "text": "Vector databases store and index high-dimensional vectors for fast approximate nearest-neighbour search.",
        "source": "blog",
        "category": "ai",
        "year": 2023,
    },
    {
        "id": 9,
        "text": "BM25 is a probabilistic ranking function used in information retrieval. It scores documents based on term frequency and inverse document frequency.",
        "source": "paper",
        "category": "ai",
        "year": 2021,
    },
    {
        "id": 10,
        "text": "Cross-encoders jointly encode a query and a document together, producing a relevance score more accurate than bi-encoders.",
        "source": "paper",
        "category": "ai",
        "year": 2022,
    },
    {
        "id": 11,
        "text": "RAG combines retrieval with generation: relevant documents are fetched and injected into the LLM prompt as context.",
        "source": "paper",
        "category": "ai",
        "year": 2023,
    },
]


# ==============================================================================
# Assignment 1 — Chunking
# ==============================================================================

print("=== Assignment 1: Chunking ===")


def chunk_text(text, chunk_size=10, overlap=3):
    """
    Sliding-window word-level chunker.

    chunk_size: max words per chunk
    overlap:    words shared between adjacent chunks (prevents boundary loss)
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap  # slide forward, keeping `overlap` words
    return chunks


long_doc = (
    "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models "
    "by giving them access to external knowledge bases. Instead of relying solely on parametric "
    "memory baked into model weights, RAG retrieves relevant documents at inference time and "
    "injects them into the prompt. This allows the model to answer questions about private data, "
    "recent events, and domain-specific knowledge without any fine-tuning."
)

chunks = chunk_text(long_doc, chunk_size=12, overlap=3)
print(f"Original doc: {len(long_doc.split())} words")
print(f"Chunks (size=12, overlap=3): {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"  chunk[{i}]: \"{c}\"")

print(f"\nOverlap verification (last 3 words of chunk[0] == first 3 of chunk[1]):")
print(f"  end of [0]: {chunks[0].split()[-3:]}")
print(f"  start of [1]: {chunks[1].split()[:3]}")


# ==============================================================================
# Assignment 2 — Embeddings
# ==============================================================================

print("\n=== Assignment 2: Embeddings ===")

EMBEDDING_DIM = 64


def embed(text, dim=EMBEDDING_DIM):
    """
    Deterministic pseudo-embedding via hashed random projection.
    Production equivalent: sentence-transformers, text-embedding-3-small.
    """
    rng = np.random.RandomState(hash(text) % (2**31))
    vec = rng.randn(dim)
    return vec / (np.linalg.norm(vec) + 1e-9)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


doc_texts = [d["text"] for d in CORPUS]
doc_embeddings = np.array([embed(t) for t in doc_texts])

print(f"Corpus size:       {len(doc_texts)} documents")
print(f"Embedding matrix:  {doc_embeddings.shape}  (docs × dim)")
print(f"Unit norm check:   {np.allclose(np.linalg.norm(doc_embeddings, axis=1), 1.0)}")

# Show that semantically related texts have higher similarity
q_vec = embed("Python programming language")
sims = [(cosine_similarity(q_vec, doc_embeddings[i]), doc_texts[i][:55]) for i in range(len(doc_texts))]
sims.sort(reverse=True)
print("\nTop-3 by cosine to 'Python programming language':")
for s, t in sims[:3]:
    print(f"  ({s:+.4f}) {t}")


# ==============================================================================
# Assignment 3 — Vector Database
# ==============================================================================

print("\n=== Assignment 3: Vector Database ===")


class VectorDB:
    """
    Minimal in-memory vector store.
    Supports insert, cosine k-NN search, and metadata-filtered search.
    Production equivalents: Pinecone, Weaviate, pgvector, Chroma, Qdrant.
    """

    def __init__(self, dim):
        self.dim = dim
        self.vectors = []   # list of np.ndarray
        self.records = []   # list of metadata dicts

    def add(self, vector, metadata):
        assert vector.shape == (self.dim,)
        self.vectors.append(vector.copy())
        self.records.append(metadata)

    def _score_all(self, query_vec):
        matrix = np.stack(self.vectors)            # (N, dim)
        scores = matrix @ query_vec                # cosine (vecs are unit-norm)
        return scores

    def search(self, query_vec, top_k=5, filter_fn=None):
        """
        Returns top_k (score, metadata) tuples.
        filter_fn: optional callable(metadata) -> bool for pre-filtering.
        """
        scores = self._score_all(query_vec)
        indices = np.argsort(scores)[::-1]

        results = []
        for idx in indices:
            meta = self.records[idx]
            if filter_fn and not filter_fn(meta):
                continue
            results.append((float(scores[idx]), meta))
            if len(results) == top_k:
                break
        return results

    def __len__(self):
        return len(self.vectors)


# Build and populate the vector DB
vdb = VectorDB(dim=EMBEDDING_DIM)
for doc in CORPUS:
    vec = embed(doc["text"])
    vdb.add(vec, doc)

print(f"VectorDB size: {len(vdb)} documents")

query = "how do transformers process text"
q_vec = embed(query)
results = vdb.search(q_vec, top_k=3)
print(f"\nk-NN search for: '{query}'")
for score, meta in results:
    print(f"  [{meta['id']}] (sim={score:.4f}) {meta['text'][:65]}")


# ==============================================================================
# Assignment 4 — Similarity Search
# ==============================================================================

print("\n=== Assignment 4: Similarity Search ===")


def similarity_search(query_text, vdb, top_k=5):
    """Embed query, score all docs, return top-k."""
    q_vec = embed(query_text)
    return vdb.search(q_vec, top_k=top_k)


queries = [
    "When was Python first released?",
    "Tell me about the Eiffel Tower height.",
    "What is GPT-4?",
]

for q in queries:
    hits = similarity_search(q, vdb, top_k=2)
    print(f"\nQuery: '{q}'")
    for score, meta in hits:
        print(f"  [{meta['id']}] (cos={score:.4f}) [{meta['category']}] {meta['text'][:60]}")


# ==============================================================================
# Assignment 5 — Hybrid Search (BM25 + Dense)
# ==============================================================================

print("\n=== Assignment 5: Hybrid Search (BM25 + Dense) ===")


def tokenize(text):
    """Lowercase, split on whitespace, strip punctuation."""
    import re
    return re.findall(r'\b\w+\b', text.lower())


class BM25:
    """
    Okapi BM25 ranking function from scratch.

    score(q, d) = Σ IDF(t) · (tf(t,d) · (k1+1)) / (tf(t,d) + k1·(1−b + b·|d|/avgdl))

    k1 ∈ [1.2, 2.0]   — controls term-frequency saturation
    b  ∈ [0, 1]        — controls document-length normalisation (0.75 typical)
    """

    def __init__(self, corpus_texts, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.tokenized = [tokenize(t) for t in corpus_texts]
        self.N = len(self.tokenized)
        self.avgdl = sum(len(d) for d in self.tokenized) / self.N

        # document frequency: df[term] = # docs containing term
        self.df = {}
        for doc_tokens in self.tokenized:
            for term in set(doc_tokens):
                self.df[term] = self.df.get(term, 0) + 1

    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_text, doc_idx):
        q_terms = tokenize(query_text)
        doc_tokens = self.tokenized[doc_idx]
        dl = len(doc_tokens)
        tf_map = {}
        for t in doc_tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        total = 0.0
        for term in q_terms:
            tf = tf_map.get(term, 0)
            idf = self.idf(term)
            num = tf * (self.k1 + 1)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += idf * (num / denom)
        return total

    def rank(self, query_text):
        scores = [(self.score(query_text, i), i) for i in range(self.N)]
        scores.sort(reverse=True)
        return scores  # list of (score, doc_idx)


def reciprocal_rank_fusion(rankings, k=60):
    """
    RRF: fuse multiple ranked lists into one.
    score(d) = Σ_r 1 / (k + rank_r(d))
    """
    rrf_scores = {}
    for ranked_list in rankings:
        for rank, (_, doc_idx) in enumerate(ranked_list):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query_text, vdb, bm25, top_k=5):
    """Fuse dense (cosine) and sparse (BM25) rankings via RRF."""
    # Dense ranking
    q_vec = embed(query_text)
    dense_scores = vdb._score_all(q_vec)
    dense_ranked = sorted(enumerate(dense_scores), key=lambda x: x[1], reverse=True)
    dense_ranked = [(s, i) for i, s in dense_ranked]

    # Sparse (BM25) ranking
    bm25_ranked = bm25.rank(query_text)

    # Fuse
    fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked])
    return fused[:top_k]


bm25 = BM25(doc_texts)

query = "Python programming arrays numerical"
print(f"Query: '{query}'\n")

# Dense only
q_vec = embed(query)
dense = vdb.search(q_vec, top_k=5)
print("Dense (cosine) top-5:")
for score, meta in dense:
    print(f"  [{meta['id']}] (cos={score:.4f}) {meta['text'][:60]}")

# BM25 only
bm25_results = bm25.rank(query)
print("\nBM25 top-5:")
for score, idx in bm25_results[:5]:
    print(f"  [{idx}] (bm25={score:.4f}) {doc_texts[idx][:60]}")

# Hybrid
hybrid = hybrid_search(query, vdb, bm25, top_k=5)
print("\nHybrid (RRF) top-5:")
for idx, rrf_score in hybrid:
    print(f"  [{idx}] (rrf={rrf_score:.4f}) {doc_texts[idx][:60]}")


# ==============================================================================
# Assignment 6 — Metadata Filtering
# ==============================================================================

print("\n=== Assignment 6: Metadata Filtering ===")

# Pre-filter: only search within docs matching a predicate
query = "large language model released recently"
q_vec = embed(query)

print(f"Query: '{query}'")

# No filter — all docs
no_filter = vdb.search(q_vec, top_k=3)
print("\nNo filter (top-3):")
for score, meta in no_filter:
    print(f"  [{meta['id']}] cat={meta['category']} year={meta['year']} (cos={score:.4f}) {meta['text'][:55]}")

# Category filter: only 'ai' documents
ai_filter = vdb.search(q_vec, top_k=3, filter_fn=lambda m: m["category"] == "ai")
print("\nFilter: category == 'ai' (top-3):")
for score, meta in ai_filter:
    print(f"  [{meta['id']}] cat={meta['category']} year={meta['year']} (cos={score:.4f}) {meta['text'][:55]}")

# Composite filter: 'ai' category AND year >= 2023
recent_ai = vdb.search(
    q_vec, top_k=3,
    filter_fn=lambda m: m["category"] == "ai" and m["year"] >= 2023
)
print("\nFilter: category == 'ai' AND year >= 2023 (top-3):")
for score, meta in recent_ai:
    print(f"  [{meta['id']}] cat={meta['category']} year={meta['year']} (cos={score:.4f}) {meta['text'][:55]}")

# Source filter
wiki_filter = vdb.search(q_vec, top_k=3, filter_fn=lambda m: m["source"] == "wikipedia")
print("\nFilter: source == 'wikipedia' (top-3):")
for score, meta in wiki_filter:
    print(f"  [{meta['id']}] cat={meta['category']} src={meta['source']} (cos={score:.4f}) {meta['text'][:55]}")


# ==============================================================================
# Assignment 7 — Reranking (Cross-Encoder)
# ==============================================================================

print("\n=== Assignment 7: Reranking (Cross-Encoder) ===")


def cross_encode(query_text, doc_text):
    """
    Simulated cross-encoder relevance score.

    Real cross-encoder: BERT takes [CLS] query [SEP] doc [SEP] jointly,
    outputs a single relevance scalar.

    Simulation captures the same intuition: scoring depends on the
    interaction between query terms and document content — not on their
    vectors in isolation.

    Score components:
      1. Term overlap: how many query terms appear in the document
      2. Position-weighted overlap: matches near the start score higher
      3. Bigram (phrase) bonus: consecutive query terms found together
    """
    q_terms = set(tokenize(query_text))
    d_words = tokenize(doc_text)

    # 1. Exact term overlap (normalised by query length)
    overlap = sum(1 for w in d_words if w in q_terms) / (len(q_terms) + 1e-9)

    # 2. Position-weighted: earlier matches matter more
    position_score = 0.0
    for i, w in enumerate(d_words):
        if w in q_terms:
            position_score += 1.0 / (1.0 + i * 0.15)

    # 3. Bigram / phrase match bonus
    q_words = tokenize(query_text)
    doc_str = " ".join(d_words)
    phrase_bonus = 0.0
    for i in range(len(q_words) - 1):
        bigram = q_words[i] + " " + q_words[i + 1]
        if bigram in doc_str:
            phrase_bonus += 2.0

    return overlap + 0.4 * position_score + phrase_bonus


def rerank(query_text, candidates):
    """
    Take a list of (score, metadata) from first-stage retrieval,
    rerank with cross-encoder, return sorted (ce_score, metadata).
    """
    scored = []
    for _, meta in candidates:
        ce_score = cross_encode(query_text, meta["text"])
        scored.append((ce_score, meta))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


query = "how tall is the Eiffel Tower in Paris"
q_vec = embed(query)

# Stage 1: first-stage retrieval (dense, top-8 — intentionally broad)
stage1 = vdb.search(q_vec, top_k=8)
print(f"Query: '{query}'\n")
print("Stage 1 — Dense retrieval (top-8):")
for score, meta in stage1:
    print(f"  [{meta['id']}] (cos={score:.4f}) {meta['text'][:65]}")

# Stage 2: cross-encoder reranking
stage2 = rerank(query, stage1)
print("\nStage 2 — After cross-encoder reranking (top-5):")
for ce_score, meta in stage2[:5]:
    print(f"  [{meta['id']}] (ce={ce_score:.4f}) {meta['text'][:65]}")

print("\nRank changes:")
stage1_ids = [meta["id"] for _, meta in stage1]
stage2_ids = [meta["id"] for _, meta in stage2]
for new_rank, doc_id in enumerate(stage2_ids[:5]):
    old_rank = stage1_ids.index(doc_id)
    change = old_rank - new_rank
    arrow = f"↑{change}" if change > 0 else (f"↓{abs(change)}" if change < 0 else "  =")
    print(f"  [{doc_id}] {arrow}  (was rank {old_rank+1}, now rank {new_rank+1})")


# ==============================================================================
# Assignment 8 — Context Construction
# ==============================================================================

print("\n=== Assignment 8: Context Construction ===")


def build_context(docs, token_budget=200, dedup=True):
    """
    Assemble retrieved chunks into a single context string.

    token_budget: approximate word limit (words ≈ tokens for estimation)
    dedup:        drop chunks whose content overlaps significantly with
                  an already-included chunk (Jaccard ≥ 0.5)
    """

    def jaccard(text_a, text_b):
        a, b = set(tokenize(text_a)), set(tokenize(text_b))
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    included = []
    used_words = 0

    for doc in docs:
        text = doc["text"]
        words = len(text.split())

        if used_words + words > token_budget:
            # Truncate to fit budget
            remaining = token_budget - used_words
            if remaining < 10:
                break
            text = " ".join(text.split()[:remaining]) + " …"
            words = remaining

        if dedup:
            # Skip if too similar to any already-included chunk
            too_similar = any(jaccard(text, inc["text"]) >= 0.5 for inc in included)
            if too_similar:
                continue

        included.append({**doc, "text": text})
        used_words += words

    return included


def build_rag_prompt(question, context_docs):
    """Format retrieved context + question into an LLM-ready prompt."""
    if not context_docs:
        context_str = "(no relevant context found)"
    else:
        parts = []
        for i, doc in enumerate(context_docs, 1):
            source_tag = f"[{doc['source']}]"
            parts.append(f"[{i}] {source_tag} {doc['text']}")
        context_str = "\n".join(parts)

    return (
        "Answer the question based only on the context below.\n"
        "If the context does not contain the answer, say \"I don't know\".\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def rag_pipeline(question, vdb, bm25, top_k_retrieve=8, top_k_rerank=5, token_budget=150, filter_fn=None):
    """
    Full production-grade RAG pipeline:
    1. Hybrid retrieval  (BM25 + dense → RRF)
    2. Metadata filtering
    3. Cross-encoder reranking
    4. Context construction (dedup + budget)
    5. Prompt assembly
    """
    print(f"Question: {question}")

    # Step 1: Hybrid retrieval
    hybrid = hybrid_search(question, vdb, bm25, top_k=top_k_retrieve)
    candidates = [(rrf, CORPUS[idx]) for idx, rrf in hybrid]
    print(f"  Step 1 — Hybrid retrieval: {len(candidates)} candidates")

    # Step 2: Metadata filter (optional)
    if filter_fn:
        candidates = [(s, m) for s, m in candidates if filter_fn(m)]
        print(f"  Step 2 — After metadata filter: {len(candidates)} candidates")

    # Step 3: Cross-encoder reranking
    reranked = rerank(question, candidates)[:top_k_rerank]
    print(f"  Step 3 — After reranking: {len(reranked)} candidates")

    # Step 4: Context construction
    reranked_docs = [meta for _, meta in reranked]
    context_docs = build_context(reranked_docs, token_budget=token_budget)
    total_words = sum(len(d["text"].split()) for d in context_docs)
    print(f"  Step 4 — Context: {len(context_docs)} docs, ~{total_words} words")

    # Step 5: Prompt
    prompt = build_rag_prompt(question, context_docs)
    return prompt


print("─" * 60)
prompt = rag_pipeline(
    "How tall is the Eiffel Tower and where is it located?",
    vdb, bm25,
    top_k_retrieve=8, top_k_rerank=4, token_budget=120,
)
print("\nGenerated RAG prompt:")
print(prompt)

print("\n" + "─" * 60)
prompt2 = rag_pipeline(
    "What is GPT-4 and when was it released?",
    vdb, bm25,
    top_k_retrieve=8, top_k_rerank=4, token_budget=120,
    filter_fn=lambda m: m["category"] == "ai",
)
print("\nGenerated RAG prompt:")
print(prompt2)

print("\n" + "─" * 60)
print("\n=== Summary: Full Pipeline Stages ===")
print("""
  Documents
      ↓ Chunk (sliding window, overlap)
      ↓ Embed  (bi-encoder → unit vectors)
      ↓ Index  (VectorDB)
                                      Query
                                        ↓ Embed query
                                        ↓ Hybrid search  (BM25 + dense → RRF)
                                        ↓ Metadata filter
                                        ↓ Rerank          (cross-encoder)
                                        ↓ Context build   (dedup + budget)
                                        ↓ Prompt assembly
                                        ↓ LLM → Answer
""")
