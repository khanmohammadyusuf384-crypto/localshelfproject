from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
from langchain_chroma import Chroma

from localshelf_embeddings import LocalSentenceTransformerEmbeddings


BASE_DIR = Path(__file__).resolve().parent
BOOKS_WITH_EMOTIONS_PATH = BASE_DIR / "books_with_emotions.csv"
TAGGED_DESCRIPTION_PATH = BASE_DIR / "tagged_description.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"
FAVORITES_PATH = BASE_DIR / "saved_books.json"
INTERACTIONS_PATH = BASE_DIR / "ranking_interactions.jsonl"
FALLBACK_COVER = "cover-not-found.jpg"

RANKING_PROFILES = {
    "Balanced": {"semantic": 0.55, "keyword": 0.18, "mood": 0.12, "rating": 0.15},
    "Semantic Heavy": {"semantic": 0.72, "keyword": 0.10, "mood": 0.08, "rating": 0.10},
    "Keyword Heavy": {"semantic": 0.42, "keyword": 0.34, "mood": 0.09, "rating": 0.15},
    "Mood Heavy": {"semantic": 0.45, "keyword": 0.12, "mood": 0.28, "rating": 0.15},
    "Popularity Heavy": {"semantic": 0.45, "keyword": 0.13, "mood": 0.07, "rating": 0.35},
}

MOOD_TO_COLUMN = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness",
}

QUERY_MODIFIER_TERMS = {
    "hopeful",
    "tense",
    "short",
    "long",
    "dark",
    "light",
    "funny",
    "serious",
    "cozy",
    "gritty",
    "slow",
    "fast",
    "uplifting",
    "sad",
    "happy",
    "suspenseful",
    "romantic",
    "mysterious",
    "emotional",
    "adventurous",
}

MODIFIER_MAP = {
    "sad": ["tragic", "grief", "loss", "dark"],
    "hopeful": ["uplifting", "inspiring", "positive", "hope"],
    "tense": ["dark", "thrilling", "suspense", "intense"],
    "romantic": ["love", "relationship", "passion"],
    "funny": ["humor", "comedy", "lighthearted"],
    "cozy": ["small town", "gentle", "warm", "comfort"],
    "gritty": ["violent", "hard", "crime", "survival"],
    "mysterious": ["mystery", "secret", "detective", "hidden"],
    "adventurous": ["quest", "journey", "adventure", "travel"],
}


def require_project_data() -> None:
    """Check that the generated local files exist before the app starts."""
    missing = [
        str(path.name)
        for path in [BOOKS_WITH_EMOTIONS_PATH, TAGGED_DESCRIPTION_PATH]
        if not path.exists()
    ]
    if not CHROMA_DIR.exists():
        missing.append(CHROMA_DIR.name)

    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing project data: {missing_list}. Run `python build_localshelf_catalog.py` first."
        )


def load_saved_books() -> set[int]:
    """Load persisted favorites from disk."""
    if not FAVORITES_PATH.exists():
        return set()

    try:
        data = json.loads(FAVORITES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()

    values = data.get("saved_isbns", data if isinstance(data, list) else [])
    saved: set[int] = set()
    for value in values:
        try:
            saved.add(int(value))
        except (TypeError, ValueError):
            continue
    return saved


def persist_saved_books(saved: set[int]) -> None:
    """Write favorites to a small local JSON file."""
    payload = {
        "saved_isbns": sorted(saved),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    FAVORITES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_interaction(event_name: str, isbn: int, context: dict[str, Any] | None = None) -> None:
    """Append lightweight interaction events for future learned ranking work."""
    payload = {
        "event": event_name,
        "isbn13": int(isbn),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "context": context or {},
    }
    with INTERACTIONS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload) + "\n")


require_project_data()

books = pd.read_csv(BOOKS_WITH_EMOTIONS_PATH)
saved_books = load_saved_books()

books["large_thumbnail"] = books["thumbnail"].fillna("") + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna() | (books["thumbnail"].astype(str).str.len() == 0),
    FALLBACK_COVER,
    books["large_thumbnail"],
)
books["average_rating"] = pd.to_numeric(books["average_rating"], errors="coerce").fillna(0.0)
books["published_year"] = pd.to_numeric(books["published_year"], errors="coerce").fillna(0).astype(int)
books["ratings_count"] = pd.to_numeric(books["ratings_count"], errors="coerce").fillna(0).astype(int)
books["num_pages"] = pd.to_numeric(books["num_pages"], errors="coerce").fillna(0).astype(int)
books["isbn13"] = pd.to_numeric(books["isbn13"], errors="coerce").fillna(0).astype(np.int64)

db_books = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=LocalSentenceTransformerEmbeddings(),
)


def format_authors(raw_authors: str) -> str:
    authors_split = [author.strip() for author in str(raw_authors).split(";") if author.strip()]
    if len(authors_split) == 2:
        return f"{authors_split[0]} and {authors_split[1]}"
    if len(authors_split) > 2:
        return f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
    return authors_split[0] if authors_split else "Unknown author"


def truncate_description(description: str, word_limit: int = 36) -> str:
    words = str(description).split()
    if len(words) <= word_limit:
        return " ".join(words)
    return " ".join(words[:word_limit]) + "..."


def preprocess_query(query: str) -> str:
    filler_words = {"a", "an", "the", "about", "please", "recommend", "suggest", "book", "books"}
    normalized = str(query).lower().strip()
    if not normalized:
        return ""

    tokens = re.findall(r"\b[\w']+\b", normalized)
    filtered_tokens = [token for token in tokens if token not in filler_words]
    cleaned = " ".join(filtered_tokens).strip()
    return cleaned if len(cleaned) >= 2 else ""


def rewrite_conversational_query(query: str, chat_history: list[dict[str, str]] | None) -> str:
    """Use recent chat turns plus an optional LLM to turn conversation into search text."""
    raw_query = str(query or "").strip()
    if not raw_query:
        return ""

    context = " ".join(
        turn.get("content", "")
        for turn in (chat_history or [])[-4:]
        if turn.get("role") == "user"
    )
    combined = f"{context} {raw_query}".strip()

    # Optional local LLM hook: set LOCALSHELF_QUERY_LLM to a cached HF model name.
    # The deterministic fallback keeps the app free and fast by default.
    model_name = os.getenv("LOCALSHELF_QUERY_LLM", "").strip()
    if model_name:
        try:
            from transformers import pipeline

            prompt = (
                "Rewrite this reader request as a concise book search query. "
                f"Request: {combined}\nSearch query:"
            )
            generator = pipeline("text-generation", model=model_name, max_new_tokens=32)
            generated = generator(prompt)[0]["generated_text"]
            combined = generated.split("Search query:")[-1].strip()
        except Exception:
            pass

    return preprocess_query(combined)


def parse_query_structure(clean_query: str) -> dict[str, list[str]]:
    tokens = str(clean_query).split()
    if not tokens:
        return {"keywords": [], "modifiers": []}

    modifiers = [token for token in tokens if token in QUERY_MODIFIER_TERMS]
    keywords = [token for token in tokens if token not in QUERY_MODIFIER_TERMS]
    if not keywords and modifiers:
        keywords = [modifiers[-1]]
        modifiers = modifiers[:-1]
    return {"keywords": keywords, "modifiers": modifiers}


def extract_isbn_from_page_content(page_content: str) -> int | None:
    text = str(page_content).strip().strip('"')
    if not text:
        return None

    first_token = text.split()[0]
    try:
        return int(first_token)
    except (TypeError, ValueError):
        return None


def get_text_for_matching(row: pd.Series) -> str:
    return (
        str(row.get("title_and_subtitle") or row.get("title") or "")
        + " "
        + str(row.get("authors") or "")
        + " "
        + str(row.get("description") or "")
    ).lower()


def keyword_boost_for(row: pd.Series, query_terms: list[str]) -> float:
    if not query_terms:
        return 0.0

    title_text = str(row.get("title_and_subtitle") or row.get("title") or "").lower()
    description_text = str(row.get("description") or "").lower()
    title_matches = sum(1 for term in query_terms if term in title_text)
    description_matches = sum(1 for term in query_terms if term in description_text)
    raw_score = (title_matches * 0.35) + (description_matches * 0.12)
    return min(raw_score, 1.0)


def modifier_boost_for(row: pd.Series, query_modifiers: list[str]) -> float:
    if not query_modifiers:
        return 0.0

    text = get_text_for_matching(row)
    boost = 0.0
    for modifier in query_modifiers:
        related_terms = MODIFIER_MAP.get(modifier, [])
        boost += sum(1 for term in related_terms if term in text) * 0.08
    return min(boost, 1.0)


def mood_score_for(row: pd.Series, tone: str) -> float:
    column = MOOD_TO_COLUMN.get(tone)
    if not column:
        return 0.0
    return float(row.get(column, 0.0) or 0.0)


def build_explanation(row: pd.Series, query_terms: list[str], query_modifiers: list[str], tone: str) -> str:
    reasons = []
    semantic_score = float(row.get("semantic_score", 0.0) or 0.0)
    keyword_score = float(row.get("keyword_boost", 0.0) or 0.0)
    mood_score = float(row.get("mood_score", 0.0) or 0.0)

    if semantic_score:
        reasons.append(f"semantic match {semantic_score:.2f}")
    if keyword_score:
        matched = [term for term in query_terms if term in get_text_for_matching(row)]
        terms = ", ".join(matched[:4]) if matched else "query terms"
        reasons.append(f"keyword match on {terms}")
    if query_modifiers and float(row.get("modifier_boost", 0.0) or 0.0):
        reasons.append(f"style match for {', '.join(query_modifiers)}")
    if tone != "All" and mood_score:
        reasons.append(f"{tone.lower()} mood score {mood_score:.2f}")
    if float(row.get("average_rating", 0.0) or 0.0) >= 4.0:
        reasons.append(f"strong reader rating {float(row['average_rating']):.2f}")

    return "; ".join(reasons) if reasons else "browse result from your local catalog"


def apply_filters(
    book_recs: pd.DataFrame,
    category: str,
    tone: str,
    min_rating: float,
    sort_by: str,
    author: str,
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    filtered = book_recs.copy()

    if category != "All":
        filtered = filtered[filtered["simple_categories"] == category]
    filtered = filtered[filtered["average_rating"] >= min_rating]

    if author:
        filtered = filtered[filtered["authors"].str.lower().str.contains(author.lower(), na=False)]

    filtered = filtered[
        (filtered["published_year"] != 0)
        & (filtered["published_year"] >= year_min)
        & (filtered["published_year"] <= year_max)
    ]

    if sort_by == "Highest Rated":
        filtered = filtered.sort_values(by=["average_rating", "ratings_count"], ascending=[False, False])
    elif sort_by == "Newest First":
        filtered = filtered.sort_values(by=["published_year", "average_rating"], ascending=[False, False])
    elif sort_by == "Shortest Reads":
        filtered = filtered.sort_values(by=["num_pages", "average_rating"], ascending=[True, False])
    elif sort_by == "Saved First":
        filtered = filtered.sort_values(by=["saved", "final_score"], ascending=[False, False])

    return filtered


def rerank_with_composite_score(
    book_recs: pd.DataFrame,
    ordered_isbns: list[int],
    query_keywords: list[str],
    query_modifiers: list[str],
    tone: str,
    ranking_profile: str,
) -> pd.DataFrame:
    reranked = book_recs.copy()
    if reranked.empty:
        return reranked

    weights = RANKING_PROFILES.get(ranking_profile, RANKING_PROFILES["Balanced"])
    isbn_to_rank = {isbn: rank for rank, isbn in enumerate(ordered_isbns)}
    total = max(len(ordered_isbns), 1)

    def semantic_score_for(isbn: int) -> float:
        rank = isbn_to_rank.get(isbn, total)
        return max(0.0, 1.0 - (rank / total))

    query_terms = [str(term).lower() for term in query_keywords if str(term).strip()]
    reranked["semantic_score"] = reranked["isbn13"].map(semantic_score_for).astype(float)
    reranked["semantic_score"] = reranked["semantic_score"].apply(lambda value: value if value >= 0.2 else value * 0.5)
    reranked["keyword_boost"] = reranked.apply(lambda row: keyword_boost_for(row, query_terms), axis=1)
    reranked["modifier_boost"] = reranked.apply(lambda row: modifier_boost_for(row, query_modifiers), axis=1)
    reranked["mood_score"] = reranked.apply(lambda row: mood_score_for(row, tone), axis=1)
    reranked["rating_boost"] = (reranked["average_rating"] / 5.0).clip(0, 1)
    reranked["final_score"] = (
        weights["semantic"] * reranked["semantic_score"]
        + weights["keyword"] * reranked["keyword_boost"]
        + weights["mood"] * np.maximum(reranked["mood_score"], reranked["modifier_boost"])
        + weights["rating"] * reranked["rating_boost"]
    )
    reranked["explanation"] = reranked.apply(
        lambda row: build_explanation(row, query_terms, query_modifiers, tone),
        axis=1,
    )
    return reranked.sort_values(
        by=["final_score", "average_rating", "ratings_count"],
        ascending=[False, False, False],
    )


def retrieve_recommendations(
    query: str,
    category: str,
    tone: str,
    min_rating: float,
    sort_by: str,
    max_results: int,
    author: str,
    year_min: int,
    year_max: int,
    ranking_profile: str,
    chat_history: list[dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    clean_query = rewrite_conversational_query(query, chat_history)
    parsed_query = parse_query_structure(clean_query)

    if clean_query:
        recs = db_books.similarity_search(clean_query, k=80)
        ordered_isbns: list[int] = []
        for rec in recs:
            isbn = extract_isbn_from_page_content(rec.page_content)
            if isbn is not None and isbn not in ordered_isbns:
                ordered_isbns.append(isbn)

        book_recs = books.set_index("isbn13").reindex(ordered_isbns).dropna(subset=["title"]).reset_index()
        book_recs = rerank_with_composite_score(
            book_recs=book_recs,
            ordered_isbns=ordered_isbns,
            query_keywords=parsed_query["keywords"],
            query_modifiers=parsed_query["modifiers"],
            tone=tone,
            ranking_profile=ranking_profile,
        )
        mode = (
            f"Semantic search for `{clean_query}` "
            f"using {ranking_profile} ranking"
        )
    else:
        book_recs = books.copy()
        book_recs["semantic_score"] = 0.0
        book_recs["keyword_boost"] = 0.0
        book_recs["modifier_boost"] = 0.0
        book_recs["mood_score"] = book_recs.apply(lambda row: mood_score_for(row, tone), axis=1)
        book_recs["rating_boost"] = (book_recs["average_rating"] / 5.0).clip(0, 1)
        book_recs["final_score"] = book_recs["rating_boost"] + (0.15 * book_recs["mood_score"])
        book_recs["explanation"] = book_recs.apply(
            lambda row: build_explanation(row, [], parsed_query["modifiers"], tone),
            axis=1,
        )
        book_recs = book_recs.sort_values(by=["final_score", "ratings_count"], ascending=[False, False])
        mode = f"Browse mode using {ranking_profile} ranking"

    book_recs["saved"] = book_recs["isbn13"].astype(int).isin(saved_books)
    filtered = apply_filters(book_recs, category, tone, min_rating, sort_by, author, year_min, year_max)
    filtered = filtered.head(int(max_results)).copy()
    metrics = evaluate_ranking(filtered, parsed_query["keywords"], tone, ranking_profile)
    return filtered, mode, metrics


def evaluate_ranking(
    recommendations: pd.DataFrame,
    query_keywords: list[str],
    tone: str,
    ranking_profile: str,
) -> dict[str, Any]:
    if recommendations.empty:
        return {
            "profile": ranking_profile,
            "coverage": 0,
            "avg_final_score": 0.0,
            "keyword_hit_rate": 0.0,
            "mood_alignment": 0.0,
            "saved_rate": 0.0,
        }

    if query_keywords:
        hits = recommendations.apply(
            lambda row: any(term in get_text_for_matching(row) for term in query_keywords),
            axis=1,
        )
        keyword_hit_rate = float(hits.mean())
    else:
        keyword_hit_rate = 0.0

    return {
        "profile": ranking_profile,
        "coverage": int(len(recommendations)),
        "avg_final_score": float(recommendations["final_score"].mean()),
        "keyword_hit_rate": keyword_hit_rate,
        "mood_alignment": float(recommendations["mood_score"].mean()) if tone != "All" else 0.0,
        "saved_rate": float(recommendations["saved"].mean()),
        "avg_rating": float(recommendations["average_rating"].mean()),
    }


def build_summary(
    recommendations: pd.DataFrame,
    mode: str,
    category: str,
    tone: str,
    sort_by: str,
    min_rating: float,
    author: str,
    year_min: int,
    year_max: int,
    metrics: dict[str, Any],
) -> str:
    if recommendations.empty:
        return (
            "### No books matched these filters\n"
            "Try lowering the minimum rating, switching the tone, or clearing the query."
        )

    years = recommendations["published_year"].replace(0, np.nan).dropna()
    oldest = int(years.min()) if not years.empty else "Unknown"
    newest = int(years.max()) if not years.empty else "Unknown"
    return (
        "### Results\n"
        f"- Mode: {mode}\n"
        f"- Books shown: {len(recommendations)}\n"
        f"- Filters: {category}, {tone}, rating >= {min_rating:.1f}, {sort_by}\n"
        f"- Author: {author if author else 'Any'}\n"
        f"- Year range: {year_min} to {year_max}\n"
        f"- Publication span in results: {oldest} to {newest}\n"
        f"- Ranking metrics: score {metrics['avg_final_score']:.2f}, "
        f"keyword hits {metrics['keyword_hit_rate']:.0%}, saved {metrics['saved_rate']:.0%}\n"
    )


def build_metrics_markdown(metrics: dict[str, Any]) -> str:
    return (
        "### Ranking evaluation\n"
        f"- Profile: {metrics['profile']}\n"
        f"- Result coverage: {metrics['coverage']}\n"
        f"- Average final score: {metrics['avg_final_score']:.3f}\n"
        f"- Keyword hit rate: {metrics['keyword_hit_rate']:.0%}\n"
        f"- Mood alignment: {metrics['mood_alignment']:.3f}\n"
        f"- Saved-result rate: {metrics['saved_rate']:.0%}\n"
        "\nInteraction events are written to `ranking_interactions.jsonl` for future learned ranking and A/B tests."
    )


def build_table(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame(columns=["Saved", "Title", "Authors", "Rating", "Year", "Shelf", "Score", "ISBN", "Why"])

    rows = []
    for _, row in recommendations.iterrows():
        rows.append(
            {
                "Saved": "Saved" if bool(row.get("saved")) else "",
                "Title": row.get("title_and_subtitle") or row.get("title") or "Untitled",
                "Authors": format_authors(row.get("authors", "Unknown author")),
                "Rating": round(float(row.get("average_rating", 0.0)), 2),
                "Year": int(row.get("published_year", 0)),
                "Shelf": row.get("simple_categories", "Uncategorized"),
                "Score": round(float(row.get("final_score", 0.0)), 3),
                "ISBN": int(row.get("isbn13", 0)),
                "Why": row.get("explanation", ""),
            }
        )
    return pd.DataFrame(rows)


def build_book_cards(recommendations: pd.DataFrame, selected_isbn: int | None = None) -> str:
    if recommendations.empty:
        return "<div class='empty-state'>No book cards to show.</div>"

    cards = []
    for _, row in recommendations.iterrows():
        isbn = int(row.get("isbn13", 0))
        title = html.escape(str(row.get("title_and_subtitle") or row.get("title") or "Untitled"))
        authors = html.escape(format_authors(row.get("authors", "Unknown author")))
        description = html.escape(truncate_description(row.get("description", "")))
        image = html.escape(str(row.get("large_thumbnail", FALLBACK_COVER)))
        category = html.escape(str(row.get("simple_categories", "Uncategorized")))
        rating = float(row.get("average_rating", 0.0))
        year = int(row.get("published_year", 0))
        pages = int(row.get("num_pages", 0))
        selected_class = " selected-card" if selected_isbn == isbn else ""
        saved_label = "Saved" if bool(row.get("saved")) else "Save"
        explanation = html.escape(str(row.get("explanation", "")))
        year_text = str(year) if year > 0 else "Unknown year"
        pages_text = f"{pages} pages" if pages > 0 else "Page count unknown"
        cards.append(
            f"""
            <article class="book-card{selected_class}">
                <img src="{image}" alt="{title}" />
                <div class="book-card-body">
                    <div class="book-card-top">
                        <h3>{title}</h3>
                        <span class="save-pill">{saved_label}</span>
                    </div>
                    <p class="authors">by {authors}</p>
                    <p class="description">{description}</p>
                    <p class="why">{explanation}</p>
                    <div class="meta-row">
                        <span>{category}</span>
                        <span>Rating {rating:.2f}</span>
                        <span>{year_text}</span>
                        <span>{pages_text}</span>
                        <span>ISBN {isbn}</span>
                    </div>
                </div>
            </article>
            """
        )

    return "\n".join(cards)


def refresh_saved_panel() -> tuple[str, pd.DataFrame]:
    if not saved_books:
        return "<div class='empty-state'>No saved books yet.</div>", build_table(pd.DataFrame())

    saved_df = books[books["isbn13"].astype(int).isin(saved_books)].copy()
    saved_df["saved"] = True
    saved_df["semantic_score"] = 0.0
    saved_df["keyword_boost"] = 0.0
    saved_df["modifier_boost"] = 0.0
    saved_df["mood_score"] = 0.0
    saved_df["rating_boost"] = (saved_df["average_rating"] / 5.0).clip(0, 1)
    saved_df["final_score"] = saved_df["rating_boost"]
    saved_df["explanation"] = "saved to your reading list"
    saved_df = saved_df.sort_values(by=["average_rating", "ratings_count"], ascending=[False, False])
    return build_book_cards(saved_df), build_table(saved_df)


def select_result(event: gr.SelectData, records: list[dict[str, Any]] | None) -> tuple[int | None, str]:
    if not records or event.index is None:
        return None, "Select a result row to save it."

    row_index = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
    try:
        record = records[int(row_index)]
        isbn = int(record["isbn13"])
        title = record.get("title_and_subtitle") or record.get("title") or "Untitled"
    except (IndexError, KeyError, TypeError, ValueError):
        return None, "Could not read that selected row."

    return isbn, f"Selected: {title} ({isbn})"


def save_book(isbn: int | None, records: list[dict[str, Any]] | None) -> tuple[str, str, pd.DataFrame, list[dict[str, Any]], str, pd.DataFrame]:
    current = pd.DataFrame(records or [])
    if not isbn:
        saved_html, saved_table = refresh_saved_panel()
        return "Select a result first.", build_book_cards(current), build_table(current), records or [], saved_html, saved_table

    saved_books.add(int(isbn))
    persist_saved_books(saved_books)
    log_interaction("save", int(isbn))
    updated = current
    if not updated.empty:
        updated["saved"] = updated["isbn13"].astype(int).isin(saved_books)
    saved_html, saved_table = refresh_saved_panel()
    return f"Saved book {isbn}.", build_book_cards(updated, int(isbn)), build_table(updated), updated.to_dict("records"), saved_html, saved_table


def remove_saved_book(isbn: int | None, records: list[dict[str, Any]] | None) -> tuple[str, str, pd.DataFrame, list[dict[str, Any]], str, pd.DataFrame]:
    current = pd.DataFrame(records or [])
    if not isbn:
        saved_html, saved_table = refresh_saved_panel()
        return "Select a result first.", build_book_cards(current), build_table(current), records or [], saved_html, saved_table

    saved_books.discard(int(isbn))
    persist_saved_books(saved_books)
    log_interaction("remove_save", int(isbn))
    updated = current
    if not updated.empty:
        updated["saved"] = updated["isbn13"].astype(int).isin(saved_books)
    saved_html, saved_table = refresh_saved_panel()
    return f"Removed book {isbn}.", build_book_cards(updated, int(isbn)), build_table(updated), updated.to_dict("records"), saved_html, saved_table


def recommend_books(
    query: str,
    category: str,
    tone: str,
    min_rating: float,
    sort_by: str,
    max_results: int,
    author: str,
    year_min: int,
    year_max: int,
    ranking_profile: str,
    chat_history: list[dict[str, str]] | None,
) -> tuple[str, str, pd.DataFrame, list[dict[str, Any]], str, str, int | None]:
    if year_min > year_max:
        year_min, year_max = year_max, year_min

    recommendations, mode, metrics = retrieve_recommendations(
        query=query,
        category=category,
        tone=tone,
        min_rating=min_rating,
        sort_by=sort_by,
        max_results=max_results,
        author=author,
        year_min=int(year_min),
        year_max=int(year_max),
        ranking_profile=ranking_profile,
        chat_history=chat_history,
    )
    summary = build_summary(
        recommendations,
        mode,
        category,
        tone,
        sort_by,
        float(min_rating),
        author,
        int(year_min),
        int(year_max),
        metrics,
    )
    return (
        summary,
        build_book_cards(recommendations),
        build_table(recommendations),
        recommendations.to_dict("records"),
        build_metrics_markdown(metrics),
        "Select a row to save or remove it.",
        None,
    )


def chat_recommend(
    message: str,
    history: list[dict[str, str]] | None,
    category: str,
    tone: str,
    min_rating: float,
    sort_by: str,
    max_results: int,
    author: str,
    year_min: int,
    year_max: int,
    ranking_profile: str,
) -> tuple[
    list[dict[str, str]],
    str,
    str,
    str,
    pd.DataFrame,
    list[dict[str, Any]],
    str,
    str,
    int | None,
    str,
    pd.DataFrame,
    str,
]:
    history = history or []
    if not message.strip():
        empty_table = build_table(pd.DataFrame())
        return history, "", "", "", empty_table, [], "", "Send a search message first.", None, "", empty_table, ""

    new_history = history + [{"role": "user", "content": message}]
    summary, cards, table, records, metrics, selection, selected = recommend_books(
        message,
        category,
        tone,
        min_rating,
        sort_by,
        max_results,
        author,
        year_min,
        year_max,
        ranking_profile,
        new_history,
    )
    response = "I searched the catalog and ranked the strongest matches below."
    new_history = new_history + [{"role": "assistant", "content": response}]
    return new_history, "", summary, cards, table, records, metrics, selection, selected, summary, table, cards


categories = ["All"] + sorted(books["simple_categories"].fillna("Uncategorized").unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
sort_modes = ["Semantic Match", "Highest Rated", "Newest First", "Shortest Reads", "Saved First"]
year_min_value = int(books["published_year"].replace(0, np.nan).min())
year_max_value = int(books["published_year"].max())

CSS = """
.gradio-container { max-width: 1180px !important; }
.app-title { margin-bottom: 6px; }
.book-card {
    display: flex;
    gap: 16px;
    padding: 14px;
    border: 1px solid #d6dde3;
    border-radius: 8px;
    background: #ffffff;
    margin-bottom: 12px;
}
.book-card.selected-card { border-color: #2f6f6d; box-shadow: 0 0 0 2px rgba(47,111,109,.14); }
.book-card img {
    width: 104px;
    height: 152px;
    object-fit: cover;
    border-radius: 6px;
    background: #eef1f4;
    flex: 0 0 auto;
}
.book-card-body { min-width: 0; flex: 1; }
.book-card-top { display: flex; gap: 12px; align-items: flex-start; justify-content: space-between; }
.book-card h3 { font-size: 1rem; line-height: 1.25; margin: 0; color: #1e2b32; }
.authors { margin: 4px 0 0; color: #52616a; }
.description { margin: 10px 0 0; color: #2f3c43; line-height: 1.45; }
.why { margin: 10px 0 0; color: #365f5d; font-size: .9rem; font-weight: 600; }
.meta-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; font-size: .82rem; }
.meta-row span, .save-pill {
    background: #edf4f2;
    border: 1px solid #d5e4df;
    color: #254441;
    padding: 3px 8px;
    border-radius: 999px;
}
.empty-state {
    border: 1px dashed #b8c3ca;
    border-radius: 8px;
    padding: 20px;
    color: #58666e;
    background: #f8fafb;
}
@media (max-width: 640px) {
    .book-card { align-items: flex-start; gap: 12px; }
    .book-card img { width: 82px; height: 122px; }
}
"""

with gr.Blocks(title="LocalShelf Explorer") as dashboard:
    result_records = gr.State([])
    selected_isbn = gr.State(None)

    gr.Markdown(
        """
        # LocalShelf Explorer
        Local-first book recommendations with semantic search, persistent favorites, ranking explanations, and conversational search.
        """,
        elem_classes=["app-title"],
    )

    with gr.Row():
        with gr.Column(scale=3):
            user_query = gr.Textbox(
                label="Search",
                placeholder="Try: a hopeful fantasy adventure with a strong emotional core",
            )
        with gr.Column(scale=1):
            submit_button = gr.Button("Explore books", variant="primary")

    with gr.Accordion("Filters and ranking", open=True):
        with gr.Row():
            category_dropdown = gr.Dropdown(choices=categories, label="Shelf", value="All")
            tone_dropdown = gr.Dropdown(choices=tones, label="Mood", value="All")
            min_rating = gr.Slider(minimum=0.0, maximum=5.0, value=3.5, step=0.1, label="Minimum rating")
        with gr.Row():
            author_input = gr.Textbox(label="Author", placeholder="Any author")
            sort_dropdown = gr.Dropdown(choices=sort_modes, label="Sort", value="Semantic Match")
            ranking_profile = gr.Dropdown(choices=list(RANKING_PROFILES), label="Ranking profile", value="Balanced")
        with gr.Row():
            year_min = gr.Slider(minimum=year_min_value, maximum=year_max_value, value=year_min_value, step=1, label="Published from")
            year_max = gr.Slider(minimum=year_min_value, maximum=year_max_value, value=year_max_value, step=1, label="Published to")
            max_results = gr.Slider(minimum=4, maximum=24, value=8, step=2, label="Results")

    with gr.Tab("Explore"):
        summary_output = gr.Markdown()
        result_table = gr.Dataframe(
            headers=["Saved", "Title", "Authors", "Rating", "Year", "Shelf", "Score", "ISBN", "Why"],
            datatype=["str", "str", "str", "number", "number", "str", "number", "number", "str"],
            interactive=False,
            wrap=True,
            label="Click a row, then save or remove it",
        )
        with gr.Row():
            selection_status = gr.Markdown("Select a row to save or remove it.")
            save_button = gr.Button("Save selected", variant="primary")
            remove_button = gr.Button("Remove selected")
        cards_output = gr.HTML()

    with gr.Tab("Favorites"):
        saved_cards_output = gr.HTML()
        saved_table_output = gr.Dataframe(interactive=False, wrap=True, label="Saved books")
        refresh_saved_btn = gr.Button("Refresh favorites")

    with gr.Tab("Conversational search"):
        chatbot = gr.Chatbot(label="Book search chat")
        chat_input = gr.Textbox(label="Message", placeholder="I want something like a cozy mystery but not too dark")
        chat_button = gr.Button("Search from conversation", variant="primary")
        chat_summary_output = gr.Markdown()
        chat_result_table = gr.Dataframe(
            headers=["Saved", "Title", "Authors", "Rating", "Year", "Shelf", "Score", "ISBN", "Why"],
            datatype=["str", "str", "str", "number", "number", "str", "number", "number", "str"],
            interactive=False,
            wrap=True,
            label="Conversation results",
        )
        chat_cards_output = gr.HTML()

    with gr.Tab("Ranking lab"):
        metrics_output = gr.Markdown()

    common_inputs = [
        user_query,
        category_dropdown,
        tone_dropdown,
        min_rating,
        sort_dropdown,
        max_results,
        author_input,
        year_min,
        year_max,
        ranking_profile,
        chatbot,
    ]

    submit_button.click(
        fn=recommend_books,
        inputs=common_inputs,
        outputs=[summary_output, cards_output, result_table, result_records, metrics_output, selection_status, selected_isbn],
    )

    result_table.select(
        fn=select_result,
        inputs=[result_records],
        outputs=[selected_isbn, selection_status],
    )

    save_button.click(
        fn=save_book,
        inputs=[selected_isbn, result_records],
        outputs=[selection_status, cards_output, result_table, result_records, saved_cards_output, saved_table_output],
    )

    remove_button.click(
        fn=remove_saved_book,
        inputs=[selected_isbn, result_records],
        outputs=[selection_status, cards_output, result_table, result_records, saved_cards_output, saved_table_output],
    )

    refresh_saved_btn.click(
        fn=refresh_saved_panel,
        inputs=[],
        outputs=[saved_cards_output, saved_table_output],
    )

    chat_button.click(
        fn=chat_recommend,
        inputs=[
            chat_input,
            chatbot,
            category_dropdown,
            tone_dropdown,
            min_rating,
            sort_dropdown,
            max_results,
            author_input,
            year_min,
            year_max,
            ranking_profile,
        ],
        outputs=[
            chatbot,
            chat_input,
            summary_output,
            cards_output,
            result_table,
            result_records,
            metrics_output,
            selection_status,
            selected_isbn,
            chat_summary_output,
            chat_result_table,
            chat_cards_output,
        ],
    )

    dashboard.load(
        fn=refresh_saved_panel,
        inputs=[],
        outputs=[saved_cards_output, saved_table_output],
    )


if __name__ == "__main__":
    dashboard.launch(theme=gr.themes.Soft(), css=CSS)
