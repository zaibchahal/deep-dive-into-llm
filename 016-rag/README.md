# 016 — RAG (Retrieval-Augmented Generation)

## Goal

By the end of this module, you should be able to answer:

* What is RAG and why is it needed?
* What is a vector database?
* How does retrieval work?
* How is retrieved context injected into the prompt?
* When do you use RAG vs fine-tuning?

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
Search document store
     ↓
Retrieve relevant documents
     ↓
Inject into prompt
     ↓
LLM generates answer based on retrieved context
```

---

## 3. RAG Pipeline

```
Offline (indexing):
  Documents → Chunk → Embed → Store in vector DB

Online (retrieval):
  Query → Embed query → Find similar chunks → Build prompt → LLM
```

---

## 4. Embeddings for Retrieval

Each document chunk is converted to a vector.

Similar chunks have similar vectors.

Search = find the chunk vectors closest to the query vector.

---

## 5. Cosine Similarity Search

Given query vector q:

```
For each document d:
    sim = cosine_similarity(q, d)

Return top-K documents by similarity
```

---

## 6. Prompt Template

```
Answer the question based only on the following context:

Context:
{retrieved_documents}

Question: {user_question}

Answer:
```

---

## 7. RAG vs Fine-Tuning

| | RAG | Fine-Tuning |
|-|-----|-------------|
| Updates knowledge | At query time | Requires retraining |
| Cost | Low (no training) | High (GPU, time) |
| Facts accuracy | High (exact text) | Can hallucinate |
| Behavior change | No | Yes |

Use RAG for: up-to-date facts, private docs, citations.

Use fine-tuning for: style, tone, domain expertise.

---

## 8. Chunking

Long documents must be split into chunks.

```
Document → 512-token chunks with 50-token overlap
```

Overlap preserves context across chunk boundaries.

---

# Coding Assignments

## Assignment 1 — Document Store

Create a simple document store:

```python
docs = [
    "Python was created by Guido van Rossum in 1991.",
    "The Eiffel Tower is 330 meters tall.",
    "GPT-4 was released by OpenAI in 2023.",
    ...
]
```

---

## Assignment 2 — Embed Documents

Create fake embeddings (random vectors) for each document.

In real RAG: use `text-embedding-3-small` or `sentence-transformers`.

---

## Assignment 3 — Retrieval Function

```python
def retrieve(query_embedding, doc_embeddings, top_k=3):
    pass
```

Return top-K most similar documents.

---

## Assignment 4 — Build RAG Prompt

```python
def build_prompt(question, retrieved_docs):
    pass
```

---

## Assignment 5 — Full RAG Pipeline

Combine retrieval + prompt building + (simulated) LLM response.

---

# Success Criteria

* Understand why RAG exists
* Implement cosine similarity retrieval
* Build a complete RAG pipeline
* Know when to use RAG vs fine-tuning
