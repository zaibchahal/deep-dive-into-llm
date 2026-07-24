import numpy as np
from main import LoRALayer

print("Running tests for 015-lora...")

np.random.seed(0)

d_in, d_out = 16, 16
W = np.random.randn(d_out, d_in)
r = 4
lora = LoRALayer(W, r=r, alpha=float(r))

# Test output shape
x = np.random.randn(5, d_in)
out = lora.forward(x)
assert out.shape == (5, d_out), f"Expected (5,{d_out}), got {out.shape}"

# Test B initialized to zeros → LoRA output equals base output at init
base_out = x @ W.T
assert np.allclose(out, base_out, atol=1e-10), "At init (B=0), LoRA must equal base model"

# Test param count
params = lora.param_count()
expected = r * d_in + d_out * r
assert params == expected, f"Expected {expected}, got {params}"

# Test W is not changed during construction
assert np.allclose(lora.W, W)

print("All tests passed.")
