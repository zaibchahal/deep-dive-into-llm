import numpy as np
from main import inference, inference_with_cache, inference_without_cache

print("Running tests for 014-inference...")

np.random.seed(0)

# Test inference returns more tokens than prompt
prompt = [3, 7, 2]
result = inference(prompt, max_new_tokens=5)
assert len(result) > len(prompt), "Inference must generate new tokens"
assert result[:len(prompt)] == prompt, "Prompt must be preserved"

# Test KV cache and no-cache produce tokens of same length
n_steps = 5
r1 = inference_without_cache(prompt, n_steps)
r2 = inference_with_cache(prompt, n_steps)
assert len(r1) == len(r2) == len(prompt) + n_steps

print("All tests passed.")
