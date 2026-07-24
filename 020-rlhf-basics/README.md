# 020 — RLHF Basics

## Goal

By the end of this module, you should be able to answer:

* What is RLHF and why is it needed?
* What is a reward model?
* What is PPO and how is it used in RLHF?
* What is DPO and how does it differ from PPO-based RLHF?
* What are the dangers of reward hacking?

---

# Theory

## 1. Why RLHF?

SFT teaches the model to imitate demonstrations.

But human preferences are complex:

* "I want a helpful answer, not just a correct one."
* "Don't be overly verbose."
* "Don't make things up."
* "Be honest even if the answer is uncomfortable."

It's hard to write examples that capture all these preferences.

RLHF lets humans **compare** responses instead of writing perfect ones.

---

## 2. The RLHF Pipeline

### Step 1: Supervised Fine-Tuning (SFT)

Already done in Module 019.

---

### Step 2: Train a Reward Model

Collect **preference data**:

```
Prompt: "Explain quantum computing."

Response A: (long, detailed, accurate)
Response B: (short, vague, partially wrong)

Human prefers: A
```

Train a model to predict which response a human would prefer:

```
reward(prompt, response) → scalar score
```

---

### Step 3: RL Training (PPO)

Use the reward model to provide a signal.

The LLM generates responses.

The reward model scores them.

PPO updates the LLM to maximize the reward.

```
for each batch:
    generate responses with LLM
    score with reward model
    compute PPO loss
    update LLM weights
```

---

## 3. Reward Model

A neural network (often another LLM with a classification head):

```
Input: (prompt, response)
Output: scalar score (higher = better)
```

Trained on comparison data:

```
Given (prompt, response_A, response_B):
  P(A preferred) = sigmoid(reward(A) - reward(B))
```

Loss:

```
loss = -log(sigmoid(r_A - r_B))
```

Where r_A > r_B when A is preferred.

---

## 4. PPO (Proximal Policy Optimization)

PPO is a reinforcement learning algorithm.

In RLHF:

* **Policy**: the LLM (generates responses)
* **Reward**: the reward model score
* **KL penalty**: prevent the LLM from drifting too far from the SFT model

Full objective:

```
maximize: E[reward(response)] - β × KL(LLM || SFT_model)
```

The KL penalty keeps the model from "reward hacking".

---

## 5. Reward Hacking

The LLM finds ways to get high reward without being genuinely good.

Example:

```
Reward model trained to prefer long answers.

LLM learns: repeat the question 10 times → high reward.
```

The KL penalty limits how far the model drifts from SFT.

---

## 6. DPO (Direct Preference Optimization)

Simpler alternative to PPO (2023).

Skips the reward model entirely.

Trains directly on preference pairs using a special loss:

```
loss = -log(sigmoid(β × (log P(y_w|x) - log P(y_l|x) - (log P_ref(y_w|x) - log P_ref(y_l|x)))))
```

Where:
* y_w = preferred (winning) response
* y_l = rejected (losing) response
* P_ref = reference (SFT) model probabilities

Much simpler to implement. Now standard.

---

## 7. Modern Variants

| Method | Notes |
|--------|-------|
| RLHF + PPO | Original (InstructGPT) |
| DPO | Simpler, no reward model |
| ORPO | Even simpler loss |
| GRPO | Used in DeepSeek, reasoning models |

---

# Coding Assignments

## Assignment 1 — Preference Dataset

Create comparison pairs:

```python
preference_data = [
    {
        "prompt": "...",
        "chosen": "...",   # preferred response
        "rejected": "..."  # non-preferred response
    },
    ...
]
```

---

## Assignment 2 — Reward Model (Simplified)

Train a simple model to score responses.

Input: response embedding.
Output: scalar reward.

Train on preference pairs.

---

## Assignment 3 — DPO Loss

Implement the DPO loss:

```python
def dpo_loss(chosen_log_probs, rejected_log_probs, ref_chosen_log_probs, ref_rejected_log_probs, beta=0.1):
    pass
```

---

## Assignment 4 — Simulate RLHF Training

Show that a model can improve based on rewards.

---

# Success Criteria

* Know the 3-step RLHF pipeline
* Understand what the reward model does
* Implement DPO loss
* Know what reward hacking is and how to mitigate it
* Know the difference between PPO-RLHF and DPO
