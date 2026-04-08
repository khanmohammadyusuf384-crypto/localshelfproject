from __future__ import annotations

from pathlib import Path
import shutil
from typing import Iterable

import kagglehub
import numpy as np
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from transformers import pipeline

from localshelf_embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    LocalSentenceTransformerEmbeddings,
)


BASE_DIR = Path(__file__).resolve().parent
# Path objects are a cleaner cross-platform way to build file paths than
# manually joining strings like "folder/file.csv".
RAW_BOOKS_PATH = BASE_DIR / "books.csv"
BOOKS_CLEANED_PATH = BASE_DIR / "books_cleaned.csv"
BOOKS_WITH_EMOTIONS_PATH = BASE_DIR / "books_with_emotions.csv"
TAGGED_DESCRIPTION_PATH = BASE_DIR / "tagged_description.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"

CATEGORY_MAPPING = {
    "Fiction": "Fiction",
    "Juvenile Fiction": "Children's Fiction",
    "Biography & Autobiography": "Nonfiction",
    "History": "Nonfiction",
    "Literary Criticism": "Nonfiction",
    "Philosophy": "Nonfiction",
    "Religion": "Nonfiction",
    "Comics & Graphic Novels": "Fiction",
    "Drama": "Fiction",
    "Juvenile Nonfiction": "Children's Nonfiction",
    "Science": "Nonfiction",
    "Poetry": "Fiction",
}
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]


def download_source_books() -> pd.DataFrame:
    """Download the source books dataset and load it into a dataframe."""
    dataset_dir = Path(kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata"))
    return pd.read_csv(dataset_dir / "books.csv")


def load_or_download_source_books() -> pd.DataFrame:
    """Use a local books file when it exists, otherwise download and save it."""
    if RAW_BOOKS_PATH.exists():
        print(f"Using local source dataset: {RAW_BOOKS_PATH.name}")
        return pd.read_csv(RAW_BOOKS_PATH)

    print("Downloading source dataset from Kaggle...")
    books = download_source_books()
    books.to_csv(RAW_BOOKS_PATH, index=False)
    print(f"Saved local source dataset: {RAW_BOOKS_PATH.name}")
    return books


def build_clean_books(books: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset and keep the fields needed by the app."""
    # `.copy()` avoids pandas "view vs copy" issues when we edit the filtered data later.
    filtered = books[
        ~(books["description"].isna())
        & ~(books["num_pages"].isna())
        & ~(books["average_rating"].isna())
        & ~(books["published_year"].isna())
    ].copy()

    filtered["words_in_description"] = filtered["description"].str.split().str.len()
    filtered = filtered[filtered["words_in_description"] >= 25].copy()
    title_subtitle = filtered[["title", "subtitle"]].fillna("").astype(str)
    # Keep subtitle text when it exists, but avoid leaving a trailing ":" for books
    # that do not actually have a subtitle.
    # `np.where(condition, a, b)` works like a vectorized if/else over a whole column.
    filtered["title_and_subtitle"] = np.where(
        filtered["subtitle"].isna(),
        filtered["title"],
        # `agg(..., axis=1)` combines values across columns for each row.
        title_subtitle.agg(": ".join, axis=1).str.rstrip(": "),
    )
    # Prefixing the description with ISBN lets us recover the matching book later
    # from raw vector-search results without storing extra metadata in Chroma.
    filtered["tagged_description"] = filtered[["isbn13", "description"]].astype(str).agg(" ".join, axis=1)
    cleaned = filtered.drop(
        ["subtitle", "words_in_description"],
        axis=1,
        errors="ignore",
    ).reset_index(drop=True)
    cleaned.to_csv(BOOKS_CLEANED_PATH, index=False)
    return cleaned


def infer_simple_category(raw_category: str | float) -> str:
    """Map detailed category text into a smaller set of shelf labels."""
    if pd.isna(raw_category):
        return "Uncategorized"

    raw_category = str(raw_category).strip()
    if raw_category in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[raw_category]

    lower = raw_category.lower()
    if "juvenile" in lower and "fiction" in lower:
        return "Children's Fiction"
    if "juvenile" in lower and ("nonfiction" in lower or "non-fiction" in lower):
        return "Children's Nonfiction"
    # `any(...)` returns True as soon as one matching token is found.
    if "fiction" in lower or any(
        token in lower
        for token in ["fantasy", "romance", "poetry", "drama", "comics", "graphic novel", "thriller", "mystery"]
    ):
        return "Fiction"
    if any(
        token in lower
        for token in [
            "history",
            "biography",
            "autobiography",
            "philosophy",
            "religion",
            "science",
            "self-help",
            "travel",
            "business",
            "psychology",
            "essay",
            "criticism",
            "reference",
            "cook",
            "health",
            "nonfiction",
            "non-fiction",
        ]
    ):
        return "Nonfiction"
    return "Uncategorized"


def add_simple_categories(books: pd.DataFrame) -> pd.DataFrame:
    """Add the simplified category column used by the UI filters."""
    books = books.copy()
    books["simple_categories"] = books["categories"].apply(infer_simple_category)
    return books


def calculate_max_emotion_scores(predictions: Iterable[list[dict[str, float]]]) -> dict[str, float]:
    """Keep the strongest score for each emotion across all sentences."""
    # Type hints like `Iterable[list[dict[str, float]]]` do not change runtime
    # behavior; they just make the expected shape of the data easier to read.
    per_emotion_scores = {label: [] for label in EMOTION_LABELS}
    for prediction in predictions:
        # The classifier returns all labels for each sentence. Sorting keeps the
        # label order stable so we can aggregate sentence-level scores safely.
        # `key=lambda item: item["label"]` tells `sorted()` which value to sort by.
        sorted_predictions = sorted(prediction, key=lambda item: item["label"])
        for index, label in enumerate(EMOTION_LABELS):
            # `enumerate(...)` gives both the index and the value while looping.
            per_emotion_scores[label].append(sorted_predictions[index]["score"])
    return {
        label: float(np.max(scores)) if scores else 0.0
        for label, scores in per_emotion_scores.items()
    }


def add_emotion_scores(books: pd.DataFrame) -> pd.DataFrame:
    """Run the emotion model on each description and save the scores."""
    books = books.copy()
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        # `top_k=None` asks transformers to return scores for every label, not just the best one.
        top_k=None,
        # `device=-1` means run on CPU.
        device=-1,
    )

    emotion_scores = {label: [] for label in EMOTION_LABELS}
    for description in books["description"].fillna(""):
        sentences = [sentence.strip() for sentence in str(description).split(".") if sentence.strip()]
        if not sentences:
            sentences = [""]
        predictions = classifier(sentences)
        # We keep the strongest score seen across the description so a single
        # high-signal sentence can still influence later mood filtering.
        max_scores = calculate_max_emotion_scores(predictions)
        for label in EMOTION_LABELS:
            emotion_scores[label].append(max_scores[label])

    for label in EMOTION_LABELS:
        books[label] = emotion_scores[label]

    books.to_csv(BOOKS_WITH_EMOTIONS_PATH, index=False)
    return books


def write_tagged_descriptions(books: pd.DataFrame) -> None:
    """Write one searchable text line per book for vector-store building."""
    tagged_lines = books["tagged_description"].fillna("").astype(str)
    with TAGGED_DESCRIPTION_PATH.open("w", encoding="utf-8") as file:
        for line in tagged_lines:
            file.write(line.replace("\r", " ").replace("\n", " ").strip())
            file.write("\n")


def build_vector_store(model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
    """Create the local Chroma database from the tagged descriptions file."""
    documents = []
    with TAGGED_DESCRIPTION_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if text:
                # LangChain's Chroma wrapper stores `Document` objects rather than raw strings.
                documents.append(Document(page_content=text))

    if CHROMA_DIR.exists():
        # Rebuild from scratch so the vector store always matches the latest
        # cleaned CSV and tagged descriptions on disk.
        shutil.rmtree(CHROMA_DIR)

    embedding = LocalSentenceTransformerEmbeddings(model_name=model_name)
    Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=str(CHROMA_DIR),
    )


def main() -> None:
    """Run the full local data-preparation flow for the app."""
    if BOOKS_CLEANED_PATH.exists():
        print(f"Using existing cleaned data: {BOOKS_CLEANED_PATH.name}")
        books = pd.read_csv(BOOKS_CLEANED_PATH)
    else:
        books = load_or_download_source_books()
        books = build_clean_books(books)
    books = add_simple_categories(books)
    books = add_emotion_scores(books)
    write_tagged_descriptions(books)
    build_vector_store()
    print("Local data preparation complete.")
    print(f"Created: {BOOKS_CLEANED_PATH.name}, {BOOKS_WITH_EMOTIONS_PATH.name}, {TAGGED_DESCRIPTION_PATH.name}")
    print(f"Persisted vector store: {CHROMA_DIR}")
    print("Run the app with: python localshelf_explorer.py")


if __name__ == "__main__":
    main()
