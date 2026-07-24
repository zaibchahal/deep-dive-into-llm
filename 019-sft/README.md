# 019 — SFT (Supervised Fine-Tuning)

## Goal

By the end of this module, you should be able to answer:

* What is supervised fine-tuning (SFT)?
* How does SFT differ from pretraining?
* What does instruction-response data look like?
* What is a chat template?
* What are the risks of SFT?

---

# Theory

## 1. Pretraining vs SFT

### Pretraining

```
Objective: predict next token on internet text
Data: terabytes of raw text (books, web, code)
Goal: learn language, facts, reasoning
Result: a base model (not an assistant)
```

### SFT (Supervised Fine-Tuning)

```
Objective: predict next token on instruction-response pairs
Data: thousands to millions of (instruction, response) examples
Goal: teach the model to follow instructions
Result: an assistant model
```

---

## 2. What Changes?

After pretraining:

```
Input: "The capital of France is"
Output: "Paris. It was..."   ← continues text
```

After SFT:

```
Input: "What is the capital of France?"
Output: "The capital of France is Paris."  ← answers the question
```

Same model, different behavior.

---

## 3. SFT Data Format

### Simple instruction-response

```json
{
  "instruction": "Translate 'hello' to French.",
  "response": "Bonjour."
}
```

### Chat format (OpenAI style)

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"}
  ]
}
```

---

## 4. Chat Template

The chat format is converted to a flat string with special tokens:

```
<|system|>You are a helpful assistant.<|end|>
<|user|>What is 2+2?<|end|>
<|assistant|>4<|end|>
```

Different models use different templates.

---

## 5. Training Objective

Same as pretraining: cross-entropy loss.

But we only compute loss on the **assistant** tokens.

Not on the user turn or system prompt.

```
Input tokens:  [system...] [user...] [assistant...]
Loss:            ignore       ignore    COMPUTE HERE
```

---

## 6. SFT Loss Masking

```
tokens:  [sys] [user] [<|assistant|>] [response tokens] [<|end|>]
mask:     0     0     0               1 1 1 1 1 1         1
```

The model learns to produce the response.

Not to reproduce the prompt.

---

## 7. Datasets

Popular SFT datasets:

* Alpaca (52K)
* Dolly (15K)
* OpenAssistant (90K)
* FLAN (millions)
* ShareGPT (ChatGPT conversations)

---

## 8. PEFT: Fine-Tune Efficiently

Full SFT on large models is expensive.

Common approach: **LoRA + SFT**

Only train A and B matrices (from Module 015).

---

# Coding Assignments

## Assignment 1 — Instruction Dataset

Create a small dataset:

```python
dataset = [
    {"instruction": "...", "response": "..."},
    ...
]
```

At least 20 examples.

---

## Assignment 2 — Tokenize with Template

Implement a simple chat template:

```python
def apply_template(instruction, response):
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
```

---

## Assignment 3 — Create Training Pairs

Convert template to (input_ids, target_ids, mask).

Loss mask: 0 for instruction part, 1 for response part.

---

## Assignment 4 — SFT Training Step

Implement one training step with the loss mask applied.

Show that loss decreases on the dataset.

---

# Success Criteria

* Understand why SFT is needed after pretraining
* Build an instruction-response dataset
* Implement loss masking (only on response tokens)
* Understand chat templates
