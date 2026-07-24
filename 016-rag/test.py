import numpy as np
from main import cosine_similarity, retrieve, build_prompt, doc_embeddings, documents

print("Running tests for 016-rag...")

# Test cosine similarity of identical vectors = 1
a = np.array([1.0, 0.0, 0.0])
assert abs(cosine_similarity(a, a) - 1.0) < 1e-6

# Test cosine similarity of orthogonal vectors = 0
b = np.array([0.0, 1.0, 0.0])
assert abs(cosine_similarity(a, b)) < 1e-6

# Test retrieve returns correct count
q = np.ones(64) / np.sqrt(64)
results = retrieve(q, doc_embeddings, top_k=3)
assert len(results) == 3

# Test retrieve returns sorted by similarity
sims = [r[0] for r in results]
assert sims == sorted(sims, reverse=True)

# Test build_prompt contains question
prompt = build_prompt("What is AI?", ["AI is artificial intelligence."])
assert "What is AI?" in prompt
assert "AI is artificial intelligence." in prompt

print("All tests passed.")
