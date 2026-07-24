import numpy as np
import math
import sys
sys.path.insert(0, '../009-transformer-block')
from main import transformer_block, TransformerBlockParams, layer_norm

np.random.seed(42)


def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            pe[pos, i] = math.sin(angle)
            if i + 1 < d_model:
                pe[pos, i + 1] = math.cos(angle)
    return pe


# --------------------------
# Assignment 1 — Stack N Blocks
# --------------------------

def transformer_stack(x, all_params):
    for params in all_params:
        x = transformer_block(x, params)
    return x


print("=== Assignment 1: Stack N Blocks ===")

d_model = 16
d_ff = 64
h = 4
n_blocks = 6

all_params = [TransformerBlockParams(d_model, d_ff, h) for _ in range(n_blocks)]

x = np.random.randn(5, d_model)
out = transformer_stack(x, all_params)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"After {n_blocks} blocks — shape preserved: {x.shape == out.shape}")


# --------------------------
# Assignment 2 — Full Forward Pass
# --------------------------

def full_forward_pass(token_ids, embedding_matrix, all_params, gamma_final, beta_final):
    # Step 1: Embedding lookup
    x = embedding_matrix[token_ids]

    # Step 2: Positional encoding
    seq_len = len(token_ids)
    pe = positional_encoding(seq_len, embedding_matrix.shape[1])
    x = x + pe

    # Step 3: Transformer stack
    x = transformer_stack(x, all_params)

    # Step 4: Final LayerNorm
    x = layer_norm(x, gamma_final, beta_final)

    return x


print("\n=== Assignment 2: Full Forward Pass ===")

vocab_size = 50
embedding_matrix = np.random.randn(vocab_size, d_model) * 0.02

gamma_final = np.ones(d_model)
beta_final = np.zeros(d_model)

token_ids = [3, 7, 2, 1, 15]
output = full_forward_pass(token_ids, embedding_matrix, all_params, gamma_final, beta_final)

print(f"Token IDs: {token_ids}")
print(f"Output shape: {output.shape}")
print(f"Output (first token): {np.round(output[0], 4)}")


# --------------------------
# Assignment 3 — Count Parameters
# --------------------------

def count_params(d_model, d_ff, h, n_blocks, vocab_size):
    d_k = d_model // h

    # Per block
    attn_params = h * 3 * (d_model * d_k) + (d_model * d_model)
    ffn_params = (d_model * d_ff) + d_ff + (d_ff * d_model) + d_model
    ln_params = 2 * 2 * d_model

    per_block = attn_params + ffn_params + ln_params

    # Embedding
    embed_params = vocab_size * d_model

    # Final LN
    final_ln = 2 * d_model

    total = n_blocks * per_block + embed_params + final_ln
    return total


print("\n=== Assignment 3: Parameter Count ===")

print("\nOur tiny model:")
tiny_total = count_params(d_model=16, d_ff=64, h=4, n_blocks=6, vocab_size=50)
print(f"  d_model={16}, d_ff={64}, h={4}, n_blocks={6}: {tiny_total:,} parameters")

print("\nGPT-2 Small (approximate):")
gpt2_total = count_params(d_model=768, d_ff=3072, h=12, n_blocks=12, vocab_size=50257)
print(f"  d_model=768, d_ff=3072, h=12, n_blocks=12: {gpt2_total:,} parameters (~{gpt2_total/1e6:.0f}M)")
