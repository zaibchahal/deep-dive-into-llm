import numpy as np

np.random.seed(42)


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / (norm + 1e-9)


# --------------------------
# Assignment 1 — Document Store
# --------------------------

print("=== Assignment 1: Document Store ===")

documents = [
    "Python was created by Guido van Rossum in 1991.",
    "The Eiffel Tower is located in Paris and is 330 meters tall.",
    "GPT-4 was released by OpenAI in March 2023.",
    "The Great Wall of China stretches over 21,000 kilometers.",
    "NumPy is a Python library for numerical computing with arrays.",
    "Machine learning is a subset of artificial intelligence.",
    "Paris is the capital city of France.",
    "Transformers use self-attention to process sequences in parallel.",
    "The Python programming language uses indentation for code blocks.",
    "Large language models are trained on billions of text documents.",
]

print(f"Document store: {len(documents)} documents")
for i, doc in enumerate(documents):
    print(f"  [{i}] {doc[:60]}...")


# --------------------------
# Assignment 2 — Embed Documents
# --------------------------

print("\n=== Assignment 2: Embed Documents ===")

embedding_dim = 64

def embed_text(text, dim=embedding_dim):
    """
    Fake embedder: in production use sentence-transformers or OpenAI embeddings.
    Here we deterministically hash the text to a stable pseudo-embedding.
    """
    rng = np.random.RandomState(hash(text) % (2**31))
    vec = rng.randn(dim)
    vec = vec / np.linalg.norm(vec)
    return vec


doc_embeddings = np.array([embed_text(doc) for doc in documents])

print(f"Document embeddings shape: {doc_embeddings.shape}")
print(f"Each embedding is a unit vector (norm ≈ 1.0):")
for i in range(3):
    print(f"  doc[{i}] norm: {np.linalg.norm(doc_embeddings[i]):.6f}")


# --------------------------
# Assignment 3 — Retrieval Function
# --------------------------

def retrieve(query_embedding, doc_embeddings, top_k=3):
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append((sim, i))
    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:top_k]


print("\n=== Assignment 3: Retrieval ===")

queries = [
    "When was Python created?",
    "How tall is the Eiffel Tower?",
    "What is machine learning?",
]

for query in queries:
    q_emb = embed_text(query)
    results = retrieve(q_emb, doc_embeddings, top_k=3)
    print(f"\nQuery: '{query}'")
    print("Retrieved documents:")
    for sim, idx in results:
        print(f"  [{idx}] (sim={sim:.4f}) {documents[idx][:70]}")


# --------------------------
# Assignment 4 — Build RAG Prompt
# --------------------------

def build_prompt(question, retrieved_docs):
    context = "\n".join(f"- {doc}" for doc in retrieved_docs)
    prompt = (
        "Answer the question based only on the following context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return prompt


print("\n=== Assignment 4: Build RAG Prompt ===")

question = "How tall is the Eiffel Tower?"
q_emb = embed_text(question)
results = retrieve(q_emb, doc_embeddings, top_k=2)
retrieved = [documents[idx] for _, idx in results]

prompt = build_prompt(question, retrieved)
print(prompt)


# --------------------------
# Assignment 5 — Full RAG Pipeline
# --------------------------

def simulated_llm(prompt):
    """
    Simulates an LLM response.
    In production: call OpenAI API or local model.
    """
    if "330" in prompt:
        return "The Eiffel Tower is 330 meters tall."
    elif "Python" in prompt and "1991" in prompt:
        return "Python was created in 1991 by Guido van Rossum."
    elif "GPT-4" in prompt:
        return "GPT-4 was released by OpenAI in March 2023."
    else:
        return "Based on the context provided, I can answer your question."


def rag_pipeline(question, documents, doc_embeddings, top_k=3):
    print(f"Question: {question}")

    q_emb = embed_text(question)
    results = retrieve(q_emb, doc_embeddings, top_k=top_k)
    retrieved_docs = [documents[idx] for _, idx in results]

    print(f"Retrieved {len(retrieved_docs)} documents:")
    for doc in retrieved_docs:
        print(f"  - {doc[:60]}")

    prompt = build_prompt(question, retrieved_docs)
    answer = simulated_llm(prompt)

    print(f"Answer: {answer}")
    return answer


print("\n=== Assignment 5: Full RAG Pipeline ===")

test_questions = [
    "How tall is the Eiffel Tower?",
    "When was Python invented?",
    "Tell me about GPT-4.",
]

for q in test_questions:
    print("\n" + "─" * 50)
    rag_pipeline(q, documents, doc_embeddings, top_k=2)
