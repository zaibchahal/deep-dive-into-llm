# 002 — Embeddings

## References

* [WordCanvas3D — Embedding Visualizer](https://wordcanvas3d.vercel.app/embedding) — Visualize word embeddings in 3D space using PCA or UMAP. Great for building intuition around how words cluster by meaning.

---

## Goal

Understand:

* Why token IDs cannot go directly into a Transformer
* How words/tokens become vectors
* What embeddings represent
* How similarity works
* How embedding layers are trained

---

# 1. Problem: Token IDs have no meaning

From tokenization:

Example:

```
"The cat sleeps"
```

Tokenizer:

```
The   → 15
cat   → 72
sleeps → 301
```

We now have:

```
[15, 72, 301]
```

Problem:

The model sees only numbers.

It does not know:

```
cat is an animal
cat is similar to dog
sleep is related to rest
```

Token IDs are just labels.

```
15 < 72 < 301
```

does NOT mean:

```
The < cat < sleeps
```

---

# 2. Solution: Embedding

We convert each token ID into a vector.

Example:

```
cat → [0.21, 0.54, -0.13, 0.87]
```

Now the model sees:

```
The
 ↓
[0.12, 0.44, 0.91, 0.11]


cat
 ↓
[0.21, 0.54, -0.13, 0.87]


sleeps
 ↓
[0.33, -0.21, 0.71, 0.52]
```

These vectors contain learned information.

---

# 3. Embedding Matrix

The model stores a big table.

Example:

Vocabulary size:

```
10,000 tokens
```

Embedding size:

```
4 dimensions
```

Matrix:

```
          dimensions

        4
        ↓

0   [0.12  0.43  0.55  0.21]
1   [0.11  0.31  0.72  0.18]
2   [0.98  0.12  0.44  0.65]
...
9999
```

Shape:

```
(vocabulary_size, embedding_dimension)
```

Example:

```
(10000, 4)
```

---

# 4. Lookup operation

Input:

```
token IDs:

[72, 301]
```

Embedding layer:

```
Embedding Matrix

72
 ↓

[0.21,0.54,-0.13,0.87]


301
 ↓

[0.33,-0.21,0.71,0.52]
```

Output:

```
[
 [0.21,0.54,-0.13,0.87],
 [0.33,-0.21,0.71,0.52]
]
```

---

# 5. How does the model learn embeddings?

Initially:

Random values:

```
cat:

[0.01,0.02,0.03]
```

During training:

The model predicts next tokens.

Example:

Input:

```
The cat is
```

Prediction:

```
dog
```

Wrong:

```
car
```

Backpropagation updates weights.

After billions of examples:

```
cat vector becomes closer to:

dog
kitten
animal
pet
```

and farther from:

```
car
computer
banana
```

---

# 6. Similarity Between Embeddings

We compare vectors using:

## Cosine Similarity

Formula:

```
similarity =
(A · B)
---------
|A||B|
```

Example:

```
cat:

[0.2,0.5,0.1]


dog:

[0.21,0.48,0.12]


car:

[-0.4,0.8,-0.6]
```

Result:

```
cat ↔ dog

0.92


cat ↔ car

0.15
```

Meaning:

```
cat is closer to dog
```

---

# 7. Embeddings in ChatGPT

Your message:

```
"Explain attention"
```

↓

Tokenizer:

```
[534, 921, 88]
```

↓

Embedding:

```
[
 [0.23,0.55,...],
 [0.12,0.44,...],
 [0.91,0.33,...]
]
```

↓

Transformer

↓

Output tokens.

---

# Coding Assignment 1

Create:

```
002-embeddings/

├── README.md
├── main.py
├── test.py
└── notes.md
```

---

## Assignment 1 — Build Embedding Lookup (NumPy)

Do not use PyTorch.

Create:

```python
import numpy as np


vocab_size = 10
embedding_dim = 4


embedding_matrix = ?
```

Create random embeddings:

Output:

```
Token 5:

[
0.32,
0.11,
0.55,
0.91
]
```

---

## Assignment 2 — Token to Vector

Create:

```python
def embed(token_id):
    pass
```

Example:

```python
print(embed(3))
```

Output:

```
[0.23 0.44 0.12 0.88]
```

---

## Assignment 3 — Sentence Embedding

Input:

```python
tokens = [2,5,7]
```

Output:

```
[
 vector_of_token_2,
 vector_of_token_5,
 vector_of_token_7
]
```

---

## Assignment 4 — Similarity

Implement cosine similarity:

```python
def cosine_similarity(a,b):
    pass
```

Test:

```
cat_vector
dog_vector
car_vector
```

Expected:

```
cat-dog > cat-car
```

---

# Stretch Assignment

Create a tiny embedding system:

Vocabulary:

```python
{
"cat":0,
"dog":1,
"car":2,
"king":3,
"queen":4
}
```

Convert:

```
"cat dog"
```

into:

```
[
[0.21,0.55,0.31],
[0.22,0.54,0.29]
]
```

---

After completing this, you should understand:

✅ Token IDs
✅ Embedding matrix
✅ Vector representation
✅ Similarity
✅ Why LLMs operate on vectors

Next:

**003 — Positional Encoding**

because embeddings alone don't tell the Transformer:

```
"The dog bites the man"
```

vs

```
"The man bites the dog"
```

The order information comes next.
