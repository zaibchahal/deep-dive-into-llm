import numpy as np
from main import dpo_loss, RewardModel, preference_data

print("Running tests for 020-rlhf-basics...")

# Test DPO loss: lower when chosen > rejected
loss_good = dpo_loss(chosen_log_probs=-1.0, rejected_log_probs=-5.0,
                     ref_chosen_log_probs=-2.0, ref_rejected_log_probs=-3.0, beta=0.1)
loss_bad = dpo_loss(chosen_log_probs=-5.0, rejected_log_probs=-1.0,
                    ref_chosen_log_probs=-2.0, ref_rejected_log_probs=-3.0, beta=0.1)

# When model correctly prefers chosen, loss should be lower
assert loss_good < loss_bad, f"Good loss ({loss_good:.4f}) should be < bad loss ({loss_bad:.4f})"

# Test DPO loss is positive
assert loss_good > 0
assert loss_bad > 0

# Test reward model gives higher score to chosen than rejected after training
rm = RewardModel()
for _ in range(200):
    for pair in preference_data:
        rm.train_step(pair["chosen"], pair["rejected"])

correct = 0
for pair in preference_data:
    if rm.predict(pair["chosen"]) > rm.predict(pair["rejected"]):
        correct += 1
accuracy = correct / len(preference_data)
assert accuracy >= 0.5, f"Reward model accuracy {accuracy:.2f} should be >= 0.5"

print(f"Reward model accuracy: {accuracy:.2f}")
print("All tests passed.")
