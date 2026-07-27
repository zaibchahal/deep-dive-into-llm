# 020 — RLHF Basics: Notes

## Why RLHF?

SFT models can still:
- Give plausible-sounding but wrong answers
- Be overly verbose or unhelpful
- Fail to follow nuanced preferences

RLHF aligns the model with human preferences through comparison data.

## The 3-Step Pipeline

```
1. SFT: imitate good demonstrations
2. Reward Model: learn human preferences from comparisons
3. RL (PPO or DPO): optimize LLM to maximize reward
```

## Reward Model

```
Input: (prompt, response)
Output: scalar score

Training: minimize -log(sigmoid(r_chosen - r_rejected))
```

Trained on human comparisons (A vs B).

## PPO in RLHF

```
Objective:
  maximize E[reward(response)] - β × KL(policy || reference)
```

KL penalty prevents reward hacking.

## DPO (Direct Preference Optimization)

Simpler: no reward model needed.

```
loss = -log(sigmoid(β × (log P(chosen) - log P(rejected)
                        - (log P_ref(chosen) - log P_ref(rejected)))))
```

Directly optimizes on preference pairs.

## Reward Hacking

Model finds ways to get high reward score without being genuinely good.

Example: very long responses fool a reward model that prefers length.

Prevention: KL penalty, diverse reward signals, human evaluation.

## Modern Variants

| Method | Year | Key Idea |
|--------|------|----------|
| RLHF + PPO | 2022 (InstructGPT) | Reward model + PPO |
| DPO | 2023 | Direct preference optimization |
| ORPO | 2024 | Odds ratio preference optimization |
| GRPO | 2024 | Used in DeepSeek-R1 |

## What Models Use RLHF?

- GPT-4, ChatGPT (OpenAI)
- Claude (Anthropic) — uses Constitutional AI variant
- Gemini (Google)
- LLaMA 2 Chat (Meta)

## Up Next

Module 021 — Hallucination Mitigation: self-consistency, entropy-based uncertainty, faithfulness scoring, and abstention.
