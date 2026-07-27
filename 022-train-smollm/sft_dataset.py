"""
sft_dataset.py
--------------
Loads the SFT JSONL file and formats each example into a single text
string using the model's chat template.

Why chat templates?
  Each model has a specific format for multi-turn conversations.
  SmolLM2 uses the ChatML format:

    <|im_start|>system
    You are a Google Ads AI assistant...
    <|im_end|>
    <|im_start|>user
    Write a GAQL query...
    <|im_end|>
    <|im_start|>assistant
    SELECT campaign.id...
    <|im_end|>

  Applying the template ensures the model learns the right conversation
  structure, not just raw text.

Data format (sft_examples.jsonl):
  Each line is a JSON object with a "messages" key:
  {
    "messages": [
      {"role": "system",    "content": "..."},
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
"""

import json
from pathlib import Path

from datasets import Dataset
from transformers import PreTrainedTokenizerBase


def load_sft_examples(jsonl_path: str | Path) -> list[dict]:
    """
    Load a JSONL file where each line has a "messages" key.

    Returns
    -------
    list[dict]
        Each dict has a "messages" key containing a list of role/content pairs.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"SFT data file not found: {path}")

    examples = []
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")

    print(f"[sft_dataset] loaded {len(examples)} examples from {path}")
    return examples


def build_sft_dataset(
    examples: list[dict],
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """
    Apply the tokenizer's chat template to each example and return
    a HF Dataset with a single 'text' column.

    The SFTTrainer expects a 'text' column containing the fully
    formatted string ready to tokenize.

    Parameters
    ----------
    examples : list[dict]
        Output of load_sft_examples().
    tokenizer : PreTrainedTokenizerBase
        Must have a chat_template set (SmolLM2 does by default).

    Returns
    -------
    datasets.Dataset
    """
    formatted_texts = []

    for ex in examples:
        messages = ex["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,          # return a string, not token IDs
            add_generation_prompt=False,
        )
        formatted_texts.append(text)

    dataset = Dataset.from_dict({"text": formatted_texts})
    print(f"[sft_dataset] built dataset with {len(dataset)} formatted examples")
    return dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from config import MODEL_NAME, SFT_DATA_PATH

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    examples = load_sft_examples(SFT_DATA_PATH)
    ds = build_sft_dataset(examples, tokenizer)

    print("\nSample formatted text:")
    print(ds[0]["text"][:600])
