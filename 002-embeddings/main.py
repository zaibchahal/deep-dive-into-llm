import numpy as np

np.random.seed(42)

# --------------------------
# Assignment 1 — Build Embedding Lookup (NumPy)
# --------------------------

vocab_size = 10
embedding_dim = 4

embedding_matrix = np.random.randn(vocab_size, embedding_dim)

print("=== Assignment 1: Embedding Matrix ===")
print("Shape:", embedding_matrix.shape)
print("Token 5:", embedding_matrix[5])


# --------------------------
# Assignment 2 — Token to Vector
# --------------------------

def embed(token_id):
    return embedding_matrix[token_id]


print("\n=== Assignment 2: Token to Vector ===")
print("embed(3):", embed(3))
print("embed(7):", embed(7))


# --------------------------
# Assignment 3 — Sentence Embedding
# --------------------------

def embed_sentence(token_ids):
    return np.array([embedding_matrix[tid] for tid in token_ids])


print("\n=== Assignment 3: Sentence Embedding ===")
tokens = [2, 5, 7]
sentence_embedding = embed_sentence(tokens)
print("Tokens:", tokens)
print("Embeddings:\n", sentence_embedding)


# --------------------------
# Assignment 4 — Cosine Similarity
# --------------------------

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)


print("\n=== Assignment 4: Cosine Similarity ===")

word_vocab = {"cat": 0, "dog": 1, "car": 2, "king": 3, "queen": 4}
word_embeddings = np.random.randn(5, 4)

cat_vec = word_embeddings[word_vocab["cat"]]
dog_vec = word_embeddings[word_vocab["dog"]]
car_vec = word_embeddings[word_vocab["car"]]

sim_cat_dog = cosine_similarity(cat_vec, dog_vec)
sim_cat_car = cosine_similarity(cat_vec, car_vec)

print("cat <-> dog:", round(sim_cat_dog, 4))
print("cat <-> car:", round(sim_cat_car, 4))
print("cat is closer to dog:", sim_cat_dog > sim_cat_car)


# --------------------------
# Stretch — Tiny Embedding System
# --------------------------

print("\n=== Stretch: Tiny Embedding System ===")

vocab = {
    "cat": 0,
    "dog": 1,
    "car": 2,
    "king": 3,
    "queen": 4
}

stretch_embeddings = np.random.randn(5, 3)


def embed_text(text):
    words = text.lower().split()
    vectors = []
    for word in words:
        if word in vocab:
            vectors.append(stretch_embeddings[vocab[word]])
        else:
            vectors.append(np.zeros(3))
    return np.array(vectors)


result = embed_text("cat dog")
print("Input: 'cat dog'")
print("Output:\n", result)
