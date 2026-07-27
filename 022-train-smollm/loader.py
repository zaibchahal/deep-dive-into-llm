"""
loader.py
---------
Recursively walks a directory and loads all .md, .sql, and .json files
as plain text strings.  Each file becomes one document.
"""

import json
from pathlib import Path

from config import SUPPORTED_EXTENSIONS


def _read_file(path: Path) -> str:
    """Return the text content of a single file."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()

    # Pretty-print JSON so the model sees structured key:value text
    # rather than a compacted blob.
    if path.suffix == ".json":
        try:
            text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass  # leave as-is if it's not valid JSON

    return text


def load_documents(docs_dir: str | Path) -> list[str]:
    """
    Walk *docs_dir* recursively and return every supported file as a string.

    Parameters
    ----------
    docs_dir : str or Path
        Root directory to search.

    Returns
    -------
    list[str]
        One string per file; empty files are skipped.
    """
    root = Path(docs_dir)
    if not root.exists():
        raise FileNotFoundError(f"docs_dir not found: {root}")

    documents: list[str] = []

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
            text = _read_file(path)
            if text:
                documents.append(text)

    print(f"[loader] loaded {len(documents)} documents from {root}")
    return documents


if __name__ == "__main__":
    import sys

    from config import DOCS_DIR
    target = sys.argv[1] if len(sys.argv) > 1 else DOCS_DIR
    docs = load_documents(target)
    print(f"First document preview ({len(docs[0])} chars):\n")
    print(docs[0][:500])
