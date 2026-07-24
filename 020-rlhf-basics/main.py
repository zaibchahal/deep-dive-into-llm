import numpy as np

np.random.seed(42)


# --------------------------
# Assignment 1 — Preference Dataset
# --------------------------

print("=== Assignment 1: Preference Dataset ===")

preference_data = [
    {
        "prompt": "Explain photosynthesis.",
        "chosen": "Photosynthesis is the process by which plants use sunlight, water, and CO2 to produce glucose and oxygen.",
        "rejected": "Plants make food from sunlight somehow."
    },
    {
        "prompt": "What is 15% of 80?",
        "chosen": "15% of 80 is 12. (80 × 0.15 = 12)",
        "rejected": "I think it's around 10 or 12."
    },
    {
        "prompt": "Write a polite email declining a meeting.",
        "chosen": "Dear [Name],\nThank you for the invitation. Unfortunately, I'm unable to attend on that date. Could we schedule an alternative time?\nBest regards,\n[Your Name]",
        "rejected": "Can't make it. Reschedule."
    },
    {
        "prompt": "What are the benefits of exercise?",
        "chosen": "Exercise improves cardiovascular health, builds muscle strength, boosts mood through endorphins, and reduces risk of chronic diseases.",
        "rejected": "Exercise is good for you."
    },
    {
        "prompt": "Explain what a Transformer is.",
        "chosen": "A Transformer is a neural network architecture that uses self-attention mechanisms to process sequences in parallel, introduced in the 'Attention Is All You Need' paper (2017).",
        "rejected": "It's a type of AI model."
    },
    {
        "prompt": "How do I reverse a list in Python?",
        "chosen": "You can reverse a list in Python using: `my_list.reverse()` (in-place) or `my_list[::-1]` (creates new list).",
        "rejected": "Use some Python code to do it."
    },
    {
        "prompt": "What is the capital of Australia?",
        "chosen": "The capital of Australia is Canberra.",
        "rejected": "Sydney or Melbourne, I think."
    },
    {
        "prompt": "Explain machine learning in simple terms.",
        "chosen": "Machine learning is teaching computers to learn from examples, rather than programming explicit rules. Like teaching a child to recognize cats by showing many pictures.",
        "rejected": "Machine learning is when computers do AI things."
    },
]

print(f"Preference dataset: {len(preference_data)} pairs")
print("\nSample pair:")
ex = preference_data[0]
print(f"  Prompt:   {ex['prompt']}")
print(f"  Chosen:   {ex['chosen'][:60]}...")
print(f"  Rejected: {ex['rejected']}")


# --------------------------
# Assignment 2 — Reward Model (Simplified)
# --------------------------

print("\n=== Assignment 2: Reward Model ===")

def featurize(text):
    """Simple heuristic features: length, specificity, detail."""
    words = text.split()
    return np.array([
        len(words),                                   # length
        len(text),                                    # char count
        sum(1 for c in text if c.isdigit()),         # contains numbers
        len([w for w in words if len(w) > 6]) / max(1, len(words)),  # long words ratio
        text.count('.') + text.count(','),            # punctuation (structure)
        int('.' in text[10:]) ,                      # ends with period
    ], dtype=float)


def normalize_features(features):
    max_vals = np.array([50.0, 500.0, 10.0, 1.0, 10.0, 1.0])
    return features / (max_vals + 1e-9)


class RewardModel:
    def __init__(self, n_features=6):
        self.W = np.random.randn(n_features) * 0.1
        self.b = 0.0

    def predict(self, text):
        features = normalize_features(featurize(text))
        return float(np.dot(self.W, features) + self.b)

    def train_step(self, chosen_text, rejected_text, lr=0.01):
        r_chosen = self.predict(chosen_text)
        r_rejected = self.predict(rejected_text)

        margin = r_chosen - r_rejected
        loss = -np.log(self._sigmoid(margin) + 1e-9)

        # Gradient
        grad_margin = -(1 - self._sigmoid(margin))

        f_chosen = normalize_features(featurize(chosen_text))
        f_rejected = normalize_features(featurize(rejected_text))

        grad_W = grad_margin * (f_chosen - f_rejected)
        self.W -= lr * grad_W
        self.b -= lr * grad_margin

        return loss

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


reward_model = RewardModel()

print("Training reward model on preference pairs:")
for epoch in range(100):
    total_loss = 0.0
    for pair in preference_data:
        loss = reward_model.train_step(pair["chosen"], pair["rejected"])
        total_loss += loss
    if epoch % 25 == 0 or epoch == 99:
        print(f"  Epoch {epoch:3d}: loss = {total_loss/len(preference_data):.4f}")

print("\nReward scores after training:")
for pair in preference_data[:4]:
    r_chosen = reward_model.predict(pair["chosen"])
    r_rejected = reward_model.predict(pair["rejected"])
    print(f"  Prompt: '{pair['prompt'][:40]}'")
    print(f"    Chosen:   {r_chosen:.4f}")
    print(f"    Rejected: {r_rejected:.4f}")
    print(f"    Correct:  {r_chosen > r_rejected}")


# --------------------------
# Assignment 3 — DPO Loss
# --------------------------

def dpo_loss(chosen_log_probs, rejected_log_probs, ref_chosen_log_probs, ref_rejected_log_probs, beta=0.1):
    """
    Direct Preference Optimization loss.

    chosen_log_probs:   log P(y_w | x)   — policy model on chosen
    rejected_log_probs: log P(y_l | x)   — policy model on rejected
    ref_chosen_log_probs:   log P_ref(y_w | x)   — reference model on chosen
    ref_rejected_log_probs: log P_ref(y_l | x)   — reference model on rejected
    beta: controls deviation from reference model
    """
    chosen_ratio = chosen_log_probs - ref_chosen_log_probs
    rejected_ratio = rejected_log_probs - ref_rejected_log_probs

    margin = beta * (chosen_ratio - rejected_ratio)

    sigmoid = 1 / (1 + np.exp(-np.clip(margin, -20, 20)))
    loss = -np.log(sigmoid + 1e-9)
    return float(loss)


print("\n=== Assignment 3: DPO Loss ===")

scenarios = [
    {
        "name": "Preferred response likely, rejected unlikely",
        "chosen_lp": -1.0,
        "rejected_lp": -5.0,
        "ref_chosen_lp": -2.0,
        "ref_rejected_lp": -3.0,
    },
    {
        "name": "Both equally likely (no signal)",
        "chosen_lp": -2.0,
        "rejected_lp": -2.0,
        "ref_chosen_lp": -2.0,
        "ref_rejected_lp": -2.0,
    },
    {
        "name": "Wrong: rejected more likely than chosen",
        "chosen_lp": -5.0,
        "rejected_lp": -1.0,
        "ref_chosen_lp": -2.0,
        "ref_rejected_lp": -3.0,
    },
]

for s in scenarios:
    loss = dpo_loss(s["chosen_lp"], s["rejected_lp"], s["ref_chosen_lp"], s["ref_rejected_lp"])
    print(f"\n  Scenario: {s['name']}")
    print(f"  DPO Loss: {loss:.4f}")

print("\nLower loss = model correctly assigns higher probability to chosen response.")


# --------------------------
# Assignment 4 — Simulate RLHF Training
# --------------------------

print("\n=== Assignment 4: Simulated RLHF Training ===")

# A tiny policy: maps state (prompt features) to a scalar "quality"
class TinyPolicy:
    def __init__(self, n_features=4):
        self.W = np.zeros(n_features)

    def score(self, features):
        return np.dot(self.W, features)

    def update(self, features, reward, lr=0.1):
        # Simple policy gradient: W += lr * reward * features
        self.W += lr * reward * features


policy = TinyPolicy(n_features=4)

print("Simulating RLHF with reward signal:")
for iteration in range(50):
    total_reward = 0.0
    for pair in preference_data:
        f_chosen = normalize_features(featurize(pair["chosen"]))[:4]
        f_rejected = normalize_features(featurize(pair["rejected"]))[:4]

        r_chosen = reward_model.predict(pair["chosen"])
        r_rejected = reward_model.predict(pair["rejected"])

        policy.update(f_chosen, r_chosen)
        policy.update(f_rejected, -abs(r_rejected))

        total_reward += r_chosen - r_rejected

    if iteration % 10 == 0 or iteration == 49:
        print(f"  Iteration {iteration:3d}: avg reward margin = {total_reward/len(preference_data):.4f}")

print("\nPolicy learned to prefer chosen responses over rejected ones.")
print("\nKey takeaway:")
print("  RLHF = SFT → Reward Model → PPO/DPO optimization")
print("  Modern models (ChatGPT, Claude, Gemini) all use variants of this pipeline.")
