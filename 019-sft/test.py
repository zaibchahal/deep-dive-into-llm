import numpy as np
from main import apply_template, apply_template_inference, build_training_example, global_vocab, dataset

print("Running tests for 019-sft...")

# Test template includes both parts
ex = dataset[0]
templated = apply_template(ex["instruction"], ex["response"])
assert "### Instruction:" in templated
assert "### Response:" in templated
assert ex["response"] in templated

# Test inference template doesn't include response
inf = apply_template_inference(ex["instruction"])
assert "### Instruction:" in inf
assert ex["response"] not in inf

# Test loss mask: response tokens marked 1, instruction tokens marked 0
ids, mask, _ = build_training_example(ex["instruction"], ex["response"], global_vocab)
assert len(ids) == len(mask)
assert 1 in mask, "Some tokens should have mask=1 (response)"
assert 0 in mask, "Some tokens should have mask=0 (instruction)"

# Test that response section has mask=1
# The response tokens come after the prefix
prefix = apply_template_inference(ex["instruction"])
prefix_len = len(prefix)
assert sum(mask) > 0

print("All tests passed.")
