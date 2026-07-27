"""
tokenizer_utils.py
------------------
Tokenizes a HF Dataset and groups the token stream into fixed-length
blocks so every training example has exactly `block_size` tokens.

Why block-chunking instead of per-document truncation?
  - Short docs waste context: padding inflates the batch.
  - Truncation silently drops the tail of long documents.
  - Block-chunking concatenates the entire corpus into one long token
    stream, then cuts it into equal-length windows — no waste, no loss.

The approach mirrors what Hugging Face uses in their own training scripts:
  https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""

from datasets import Dataset
from transformers import PreTrainedTokenizerBase


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int = 512,
    num_proc: int = 1,
) -> Dataset:
    """
    Tokenize every row in *dataset* then pack the tokens into
    non-overlapping blocks of *block_size*.

    Parameters
    ----------
    dataset : Dataset
        Must have a 'text' column.
    tokenizer : PreTrainedTokenizerBase
        The model's tokenizer.  Must have an EOS token.
    block_size : int
        Number of tokens per training example.
    num_proc : int
        Worker processes for .map().  Keep at 1 on MPS/Mac to avoid
        forking issues.

    Returns
    -------
    Dataset
        Columns: input_ids, attention_mask, labels.
        Every row has exactly *block_size* tokens.
    """

    # ── Step 1: tokenize each document ──────────────────────────────────
    # We add the EOS token between documents so the model learns that one
    # document ends before the next begins.
    eos = tokenizer.eos_token or ""

    def _tokenize(batch):
        texts = [t + eos for t in batch["text"]]
        return tokenizer(texts, add_special_tokens=False)

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing",
    )

    # ── Step 2: concatenate then chunk ──────────────────────────────────
    def _group_into_blocks(batch):
        # Flatten: join all sequences in the batch into one long list
        all_ids = []
        for ids in batch["input_ids"]:
            all_ids.extend(ids)

        total = len(all_ids)
        # Drop the remainder so every chunk is exactly block_size
        trimmed = total - (total % block_size)
        all_ids = all_ids[:trimmed]

        chunks = [all_ids[i : i + block_size] for i in range(0, trimmed, block_size)]

        return {
            "input_ids": chunks,
            # attention_mask is all-ones for these packed blocks
            "attention_mask": [[1] * block_size for _ in chunks],
            # For causal LM, labels == input_ids (the model predicts
            # the next token at every position)
            "labels": chunks,
        }

    blocked = tokenized.map(
        _group_into_blocks,
        batched=True,
        num_proc=num_proc,
        desc="Grouping into blocks",
    )

    print(
        f"[tokenizer_utils] {len(dataset)} docs → {len(blocked)} blocks "
        f"of {block_size} tokens each"
    )
    return blocked


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from config import BLOCK_SIZE, DOCS_DIR, MODEL_NAME
    from loader import load_documents
    from dataset import build_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    docs = load_documents(DOCS_DIR)
    ds = build_dataset(docs)
    blocked = tokenize_dataset(ds, tokenizer, block_size=BLOCK_SIZE)

    print(blocked)
    print(f"\nFirst block shape: {len(blocked[0]['input_ids'])} tokens")
    print("Decoded preview:")
    print(tokenizer.decode(blocked[0]["input_ids"][:80]))
