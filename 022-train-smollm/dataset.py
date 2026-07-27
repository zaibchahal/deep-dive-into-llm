"""
dataset.py
----------
Wraps a list of raw text strings into a Hugging Face Dataset.

Why a Dataset instead of a plain list?
  - Datasets are memory-mapped, so large corpora don't blow up RAM.
  - The HF Trainer expects a Dataset (or DatasetDict) as input.
  - It gives us .map(), .filter(), and .train_test_split() for free.
"""

from datasets import Dataset


def build_dataset(documents: list[str]) -> Dataset:
    """
    Convert a list of text strings into a HF Dataset with a single
    'text' column.

    Parameters
    ----------
    documents : list[str]
        Raw text strings from loader.load_documents().

    Returns
    -------
    datasets.Dataset
    """
    dataset = Dataset.from_dict({"text": documents})
    print(f"[dataset] created dataset with {len(dataset)} rows")
    return dataset


if __name__ == "__main__":
    from config import DOCS_DIR
    from loader import load_documents

    docs = load_documents(DOCS_DIR)
    ds = build_dataset(docs)
    print(ds)
    print("\nSample row:")
    print(ds[0]["text"][:300])
