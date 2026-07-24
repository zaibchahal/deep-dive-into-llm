import numpy as np
from main import embed, embed_sentence, cosine_similarity, embedding_matrix

print("Running tests for 002-embeddings...")

# Test embed returns correct shape
v = embed(0)
assert v.shape == (4,), f"Expected shape (4,), got {v.shape}"

# Test embed returns row from matrix
assert np.allclose(v, embedding_matrix[0])

# Test embed_sentence
result = embed_sentence([0, 1, 2])
assert result.shape == (3, 4), f"Expected shape (3, 4), got {result.shape}"

# Test cosine similarity between identical vectors is 1
a = np.array([1.0, 0.0, 0.0])
sim = cosine_similarity(a, a)
assert abs(sim - 1.0) < 1e-6, f"Expected 1.0, got {sim}"

# Test cosine similarity between orthogonal vectors is 0
b = np.array([0.0, 1.0, 0.0])
sim2 = cosine_similarity(a, b)
assert abs(sim2 - 0.0) < 1e-6, f"Expected 0.0, got {sim2}"

print("All tests passed.")
