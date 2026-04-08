from __future__ import annotations

import os
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = BASE_DIR / ".hf-cache"


class LocalSentenceTransformerEmbeddings:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """Load the local embedding model and point it at the project cache."""
        self.model_name = model_name
        # Point Hugging Face and sentence-transformers at a project-local cache
        # so setup stays self-contained instead of relying on global user folders.
        os.environ.setdefault("HF_HOME", str(LOCAL_CACHE_DIR))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(LOCAL_CACHE_DIR))
        # Offline flags make app startup fail fast if the model is missing locally
        # instead of silently trying to reach the network at runtime.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self._model = SentenceTransformer(
            model_name,
            cache_folder=str(LOCAL_CACHE_DIR),
            local_files_only=True,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Convert many texts into embedding vectors for the vector store."""
        # `List[List[float]]` means "a list of vectors", where each vector is a list of numbers.
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Convert one search query into a vector for similarity search."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()
