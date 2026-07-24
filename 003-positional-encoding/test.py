import numpy as np
import math
from main import positional_encoding

print("Running tests for 003-positional-encoding...")

# Test output shape
pe = positional_encoding(5, 8)
assert pe.shape == (5, 8), f"Expected (5, 8), got {pe.shape}"

# Test position 0 starts with sin(0) = 0
assert abs(pe[0, 0]) < 1e-9, f"PE[0,0] should be sin(0)=0, got {pe[0,0]}"

# Test position 0 second dim is cos(0) = 1
assert abs(pe[0, 1] - 1.0) < 1e-9, f"PE[0,1] should be cos(0)=1, got {pe[0,1]}"

# Test that different positions produce different vectors
pe2 = positional_encoding(10, 8)
assert not np.allclose(pe2[2], pe2[7]), "Different positions must produce different vectors"

# Test adding to embeddings
word_emb = np.ones((5, 8))
final = word_emb + pe
assert final.shape == (5, 8)

print("All tests passed.")
