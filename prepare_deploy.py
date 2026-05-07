from __future__ import annotations

import os
from pathlib import Path

from build_localshelf_catalog import CHROMA_DIR, build_vector_store


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

    sqlite_path = CHROMA_DIR / "chroma.sqlite3"
    if sqlite_path.exists():
        print(f"Using existing vector store: {sqlite_path}")
        return

    Path(CHROMA_DIR).mkdir(exist_ok=True)
    print("Building Chroma vector store for deployment...")
    build_vector_store()
    print(f"Created vector store: {sqlite_path}")


if __name__ == "__main__":
    main()
