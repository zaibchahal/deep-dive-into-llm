import numpy as np
import sys
import os

print("Running tests for 010-transformer-stack...")

np.random.seed(0)

# Add 009 to path for TransformerBlockParams
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '009-transformer-block'))
import main as block_main
TransformerBlockParams = block_main.TransformerBlockParams

# Import 010 functions directly by loading the file
import importlib.util
spec = importlib.util.spec_from_file_location("stack_main", os.path.join(os.path.dirname(__file__), "main.py"))
stack_main = importlib.util.load_from_spec = None

# Simpler: just re-execute the functions we need inline
# The 010/main.py imports from 009 via sys.path — so we just run it with the right sys.path
exec(open(os.path.join(os.path.dirname(__file__), 'main.py')).read(), globals())

d_model = 16
d_ff = 64
h = 4
n_blocks = 3

all_params = [TransformerBlockParams(d_model, d_ff, h) for _ in range(n_blocks)]

# Test stack output shape
x = np.random.randn(5, d_model)
out = transformer_stack(x, all_params)
assert out.shape == (5, d_model), f"Expected (5,{d_model}), got {out.shape}"

# Test full forward pass shape
vocab_size = 20
embedding_matrix_test = np.random.randn(vocab_size, d_model) * 0.02
gamma = np.ones(d_model)
beta = np.zeros(d_model)
token_ids = [0, 1, 2, 3]
out2 = full_forward_pass(token_ids, embedding_matrix_test, all_params, gamma, beta)
assert out2.shape == (4, d_model), f"Expected (4,{d_model}), got {out2.shape}"

print("All tests passed.")
