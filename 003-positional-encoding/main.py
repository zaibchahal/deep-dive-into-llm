import numpy as np
import math

np.random.seed(42)

# --------------------------
# Assignment 1 — Manual Positional Encoding
# --------------------------

print("=== Assignment 1: Manual Positional Encoding ===")

seq_len = 4
d_model = 4

pe_manual = np.array([
    [0,           1,            0,            1           ],
    [math.sin(1), math.cos(1), math.sin(0.1), math.cos(0.1)],
    [math.sin(2), math.cos(2), math.sin(0.2), math.cos(0.2)],
    [math.sin(3), math.cos(3), math.sin(0.3), math.cos(0.3)],
])

print("Manual PE matrix (4 positions, dim=4):")
print(np.round(pe_manual, 4))


# --------------------------
# Assignment 2 — Sinusoidal PE Function
# --------------------------

def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            pe[pos, i] = math.sin(angle)
            if i + 1 < d_model:
                pe[pos, i + 1] = math.cos(angle)
    return pe


print("\n=== Assignment 2: Sinusoidal PE Function ===")
pe = positional_encoding(6, 8)
print("PE shape:", pe.shape)
print("PE (6 positions, dim=8):")
print(np.round(pe, 4))


# --------------------------
# Assignment 3 — Add to Embeddings
# --------------------------

print("\n=== Assignment 3: Add PE to Word Embeddings ===")

word_embeddings = np.random.randn(4, 8)
pe4 = positional_encoding(4, 8)

final_input = word_embeddings + pe4

print("Word embeddings shape:", word_embeddings.shape)
print("PE shape:", pe4.shape)
print("Final input shape:", final_input.shape)
print("\nFinal input (first 2 tokens):")
print(np.round(final_input[:2], 4))


# --------------------------
# Assignment 4 — Visualize Positions Differ
# --------------------------

print("\n=== Assignment 4: Same Token, Different Position ===")

# "dog" has embedding vector
dog_embedding = np.array([0.21, 0.54, -0.13, 0.87, 0.12, -0.34, 0.67, 0.45])

pe_all = positional_encoding(10, 8)

dog_at_pos2 = dog_embedding + pe_all[2]
dog_at_pos7 = dog_embedding + pe_all[7]

print("'dog' at position 2:", np.round(dog_at_pos2, 4))
print("'dog' at position 7:", np.round(dog_at_pos7, 4))
print("Are they the same?", np.allclose(dog_at_pos2, dog_at_pos7))
print("Difference:", np.round(dog_at_pos2 - dog_at_pos7, 4))
