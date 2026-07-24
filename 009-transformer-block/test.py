import numpy as np
from main import transformer_block, TransformerBlockParams

print("Running tests for 009-transformer-block...")

np.random.seed(0)

d_model = 16
d_ff = 64
h = 4
params = TransformerBlockParams(d_model, d_ff, h)

# Test shape is preserved for single token
x1 = np.random.randn(1, d_model)
out1 = transformer_block(x1, params)
assert out1.shape == (1, d_model), f"Expected (1,{d_model}), got {out1.shape}"

# Test shape is preserved for multiple tokens
x4 = np.random.randn(6, d_model)
out4 = transformer_block(x4, params)
assert out4.shape == (6, d_model), f"Expected (6,{d_model}), got {out4.shape}"

# Test output is not identical to input (transformation happens)
assert not np.allclose(x4, out4), "Block must transform input"

print("All tests passed.")
