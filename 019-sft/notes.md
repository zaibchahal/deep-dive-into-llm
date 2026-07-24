# 019 — SFT: Notes

## Why SFT?

Base model (pretrained): continues text.
SFT model: follows instructions.

Same architecture. Different data. Different behavior.

## Pretraining vs SFT

| | Pretraining | SFT |
|-|-------------|-----|
| Data | Raw internet text | (instruction, response) pairs |
| Scale | Trillions of tokens | Thousands to millions |
| Goal | Learn language | Learn to follow instructions |
| Loss | All tokens | Response tokens only |

## Data Format

```json
{
  "instruction": "What is 2+2?",
  "response": "4"
}
```

Chat format (OpenAI):
```
system → user → assistant → user → assistant → ...
```

## Chat Template

Convert structured conversation to flat text:
```
### Instruction:
{instruction}

### Response:
{response}
```

Different models use different formats (Llama uses `[INST]`, Mistral uses its own).

## Loss Masking

Only compute loss on assistant tokens:

```
[instruction tokens] [response tokens]
      mask = 0            mask = 1
```

This teaches the model to generate responses, not to reproduce prompts.

## Common SFT Datasets

| Dataset | Size | Type |
|---------|------|------|
| Alpaca | 52K | GPT-3 distilled |
| Dolly | 15K | Human-written |
| OpenAssistant | 90K | Conversations |
| FLAN | millions | Multi-task |

## Practical Tips

- Use LoRA (module 015) for efficient fine-tuning
- Start with good base model
- Data quality > quantity
- Evaluate on held-out instructions

## Next

**RLHF** — reinforce the model with human preferences to make it safer and more helpful.
