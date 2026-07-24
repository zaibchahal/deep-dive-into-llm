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

## Congratulations!

You have completed all 20 modules:

```
001 Tokenization          → text to IDs
002 Embeddings            → IDs to vectors
003 Positional Encoding   → inject order
004 Self-Attention        → Q, K, V
005 Multi-Head Attention  → parallel heads
006 Feed-Forward Network  → per-token MLP
007 Layer Norm            → stabilize training
008 Residual Connections  → gradient flow
009 Transformer Block     → full block
010 Transformer Stack     → N blocks deep
011 Next Token Prediction → LM head
012 Decoding              → greedy / sampling
013 Train Tiny GPT        → training loop
014 Inference             → KV cache
015 LoRA                  → efficient fine-tuning
016 RAG                   → retrieval augmentation
017 Tool Calling          → LLM + functions
018 Agent                 → observe-think-act
019 SFT                   → instruction tuning
020 RLHF                  → human preference alignment
```
