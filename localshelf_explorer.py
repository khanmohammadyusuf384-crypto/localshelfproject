from pathlib import Path
import re

import gradio as gr
import numpy as np
import pandas as pd
from langchain_chroma import Chroma

from localshelf_embeddings import LocalSentenceTransformerEmbeddings


BASE_DIR = Path(__file__).resolve().parent
# Using `Path` keeps path handling readable and avoids hard-coded separators.
BOOKS_WITH_EMOTIONS_PATH = BASE_DIR / "books_with_emotions.csv"
TAGGED_DESCRIPTION_PATH = BASE_DIR / "tagged_description.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"
FALLBACK_COVER = "cover-not-found.jpg"


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


require_project_data()

books = pd.read_csv(BOOKS_WITH_EMOTIONS_PATH)
# Remote thumbnails from the source dataset can be small, so we request a larger
# variant when a cover URL exists and fall back to a local placeholder otherwise.
books["large_thumbnail"] = books["thumbnail"].fillna("") + "&fife=w800"
# `np.where(...)` is a vectorized if/else over the whole pandas column.
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna() | (books["thumbnail"].astype(str).str.len() == 0),
    FALLBACK_COVER,
    books["large_thumbnail"],
)
# `pd.to_numeric(..., errors="coerce")` turns bad values into NaN instead of crashing,
# then `fillna(...)` and `astype(...)` give us consistent numeric columns for sorting.
books["average_rating"] = pd.to_numeric(books["average_rating"], errors="coerce").fillna(0.0)
books["published_year"] = pd.to_numeric(books["published_year"], errors="coerce").fillna(0).astype(int)
books["ratings_count"] = pd.to_numeric(books["ratings_count"], errors="coerce").fillna(0).astype(int)
books["num_pages"] = pd.to_numeric(books["num_pages"], errors="coerce").fillna(0).astype(int)

db_books = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=LocalSentenceTransformerEmbeddings(),
)


def format_authors(raw_authors: str) -> str:
    """Turn a semicolon-separated author string into nicer display text."""
    authors_split = str(raw_authors).split(";")
    if len(authors_split) == 2:
        return f"{authors_split[0]} and {authors_split[1]}"
    if len(authors_split) > 2:
        return f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
    return str(raw_authors)


def truncate_description(description: str, word_limit: int = 32) -> str:
    """Shorten long descriptions so the book cards stay compact."""
    words = str(description).split()
    if len(words) <= word_limit:
        return " ".join(words)
    return " ".join(words[:word_limit]) + "..."


def preprocess_query(query: str) -> str:
    """Normalize natural-language input before semantic search."""
    filler_words = {
        "a",
        "an",
        "the",
        "about",
    }
    normalized = str(query).lower().strip()
    if not normalized:
        return ""

    # Keep only word-like tokens so punctuation doesn't leak into embeddings.
    tokens = re.findall(r"\b[\w']+\b", normalized)
    filtered_tokens = [token for token in tokens if token not in filler_words]

    cleaned = " ".join(filtered_tokens).strip()
    # Guard against very short inputs after cleanup (for example: "a", "the").
    if len(cleaned) < 2:
        return ""
    return cleaned


def parse_query_structure(clean_query: str) -> dict[str, list[str]]:
    """Split a cleaned query into core keywords and descriptive modifiers."""
    modifier_terms = {
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
    }

    tokens = str(clean_query).split()
    if not tokens:
        return {"keywords": [], "modifiers": []}

    modifiers = [token for token in tokens if token in modifier_terms]
    keywords = [token for token in tokens if token not in modifier_terms]

    # Keep at least one keyword if a user only typed descriptive language.
    if not keywords and modifiers:
        keywords = [modifiers[-1]]
        modifiers = modifiers[:-1]

    return {"keywords": keywords, "modifiers": modifiers}


def extract_isbn_from_page_content(page_content: str) -> int | None:
    """Safely extract the leading ISBN from a vector-store document."""
    text = str(page_content).strip().strip('"')
    if not text:
        return None
    
    first_token = text.split()[0]
    try:
        return int(first_token)
    except (TypeError, ValueError):
        return None


def apply_filters(
    book_recs: pd.DataFrame,
    category: str,
    tone: str,
    min_rating: float,
    sort_by: str,
) -> pd.DataFrame:
    """Apply the dropdown and slider choices to the recommended books."""
    filtered = book_recs.copy()

    if category != "All":
        filtered = filtered[filtered["simple_categories"] == category]

    filtered = filtered[filtered["average_rating"] >= min_rating]

    # Tone sorting is intentionally a ranking step, not a hard filter. We still
    # show all matching books, but mood-aligned ones rise to the top first.
    if tone == "Happy":
        filtered = filtered.sort_values(by=["joy", "average_rating"], ascending=[False, False])
    elif tone == "Surprising":
        filtered = filtered.sort_values(by=["surprise", "average_rating"], ascending=[False, False])
    elif tone == "Angry":
        filtered = filtered.sort_values(by=["anger", "average_rating"], ascending=[False, False])
    elif tone == "Suspenseful":
        filtered = filtered.sort_values(by=["fear", "average_rating"], ascending=[False, False])
    elif tone == "Sad":
        filtered = filtered.sort_values(by=["sadness", "average_rating"], ascending=[False, False])

    if sort_by == "Highest Rated":
        filtered = filtered.sort_values(by=["average_rating", "ratings_count"], ascending=[False, False])
    elif sort_by == "Newest First":
        filtered = filtered.sort_values(by=["published_year", "average_rating"], ascending=[False, False])
    elif sort_by == "Shortest Reads":
        filtered = filtered.sort_values(by=["num_pages", "average_rating"], ascending=[True, False])

    return filtered


def retrieve_recommendations(
    query: str,
    category: str,
    tone: str,
    min_rating: float,
    sort_by: str,
    max_results: int,
) -> tuple[pd.DataFrame, str]:
    """Get book recommendations from semantic search or browse mode."""
    # This return type means: the function gives back `(dataframe, message_text)`.
    query = preprocess_query(query)
    parsed_query = parse_query_structure(query)

    if query:
        recs = db_books.similarity_search(query, k=60)
        # Each vector-store document starts with the ISBN, so we try to recover it
        # safely and ignore malformed results instead of crashing the app.
        ordered_isbns = []
        for rec in recs:
            isbn = extract_isbn_from_page_content(rec.page_content)
            if isbn is not None and isbn not in ordered_isbns:
                ordered_isbns.append(isbn)

        # `set_index(...).reindex(...)` is a handy pandas pattern when you want rows
        # back in a specific custom order rather than the dataframe's default order.
        book_recs = books.set_index("isbn13").reindex(ordered_isbns).dropna(subset=["title"]).reset_index()
        mode = (
            f"Semantic match for: `{query}` "
            f"(keywords: {parsed_query['keywords']}, modifiers: {parsed_query['modifiers']})"
        )
 
    else:
        book_recs = books.copy()
        mode = "Browse mode: no query provided, showing books from your local catalog"

    filtered = apply_filters(book_recs, category, tone, min_rating, sort_by)
    return filtered.head(max_results), mode


def build_summary(recommendations: pd.DataFrame, mode: str, category: str, tone: str, sort_by: str) -> str:
    """Build the markdown summary shown above the result cards."""
    if recommendations.empty:
        return (
            "### No books matched these filters\n"
            "Try lowering the minimum rating, switching the tone, or clearing the query."
        )

    years = recommendations["published_year"].replace(0, np.nan).dropna()
    avg_rating = recommendations["average_rating"].mean()
    oldest = int(years.min()) if not years.empty else "Unknown"
    newest = int(years.max()) if not years.empty else "Unknown"

    return (
        "### LocalShelf Explorer results\n"
        f"- Mode: {mode}\n"
        f"- Books shown: {len(recommendations)}\n"
        f"- Category filter: {category}\n"
        f"- Tone filter: {tone}\n"
        f"- Sort mode: {sort_by}\n"
        f"- Average rating in results: {avg_rating:.2f}\n"
        f"- Publication span: {oldest} to {newest}"
    )


def build_book_cards(recommendations: pd.DataFrame) -> str:
    """Render the recommendation rows as HTML cards for Gradio."""
    if recommendations.empty:
        return "<div>No book cards to show.</div>"

    cards = []
    # `iterrows()` loops through dataframe rows one at a time. It is not the fastest
    # pandas tool, but it is fine here because we only render a small result set.
    for _, row in recommendations.iterrows():
        title = row.get("title_and_subtitle") or row.get("title") or "Untitled"
        authors = format_authors(row.get("authors", "Unknown author"))
        description = truncate_description(row.get("description", ""))
        image = row.get("large_thumbnail", FALLBACK_COVER)
        category = row.get("simple_categories", "Uncategorized")
        rating = float(row.get("average_rating", 0.0))
        year = int(row.get("published_year", 0))
        pages = int(row.get("num_pages", 0))
        year_text = str(year) if year > 0 else "Unknown year"
        pages_text = f"{pages} pages" if pages > 0 else "Page count unknown"

        # Gradio HTML output gives us more control over the card layout than a
        # simple dataframe or gallery component for this browse-style UI.
        cards.append(
            f"""
            <div style="display:flex; gap:16px; padding:16px; border:1px solid #d9d4c7; border-radius:18px; background:#fffaf0; margin-bottom:14px;">
                <img src="{image}" alt="{title}" style="width:110px; height:160px; object-fit:cover; border-radius:12px; background:#f3efe4;" />
                <div style="flex:1;">
                    <div style="font-size:1.05rem; font-weight:700; color:#23313f;">{title}</div>
                    <div style="margin-top:4px; color:#4e5b66;">by {authors}</div>
                    <div style="margin-top:10px; font-size:0.92rem; color:#31414d;">{description}</div>
                    <div style="margin-top:12px; display:flex; gap:10px; flex-wrap:wrap; font-size:0.85rem;">
                        <span style="background:#e8f1eb; padding:4px 10px; border-radius:999px;">{category}</span>
                        <span style="background:#edf1f7; padding:4px 10px; border-radius:999px;">Rating {rating:.2f}</span>
                        <span style="background:#f7efe1; padding:4px 10px; border-radius:999px;">{year_text}</span>
                        <span style="background:#f3e8ef; padding:4px 10px; border-radius:999px;">{pages_text}</span>
                    </div>
                </div>
            </div>
            """
        )

    return "\n".join(cards)


def recommend_books(
    query: str,
    category: str,
    tone: str,
    min_rating: float,
    sort_by: str,
    max_results: int,
):
    """Main callback used by the Explore button in the UI."""
    recommendations, mode = retrieve_recommendations(
        query=query,
        category=category,
        tone=tone,
        min_rating=min_rating,
        sort_by=sort_by,
        max_results=max_results,
    )
    summary = build_summary(recommendations, mode, category, tone, sort_by)
    cards = build_book_cards(recommendations)
    return summary, cards

# These lists are built once from the dataset and then reused by the Gradio widgets.
categories = ["All"] + sorted(books["simple_categories"].fillna("Uncategorized").unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
sort_modes = ["Semantic Match", "Highest Rated", "Newest First", "Shortest Reads"]

with gr.Blocks(theme=gr.themes.Soft()) as dashboard:
    gr.Markdown(
        """
        # LocalShelf Explorer
        A local-first book discovery app that blends semantic search with lightweight browsing filters.
        Leave the query blank if you want to explore your catalog without searching by concept.
        """
    )

    with gr.Row():
        user_query = gr.Textbox(
            label="Describe what you want to read",
            placeholder="Examples: a reflective memoir about reinvention, a tense revenge story, a hopeful short read",
        )

    with gr.Row():
        category_dropdown = gr.Dropdown(choices=categories, label="Shelf", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Mood", value="All")
        min_rating = gr.Slider(minimum=0.0, maximum=5.0, value=3.5, step=0.1, label="Minimum rating")

    with gr.Row():
        sort_dropdown = gr.Dropdown(choices=sort_modes, label="Sort results", value="Semantic Match")
        max_results = gr.Slider(minimum=4, maximum=20, value=8, step=2, label="How many books to show")
        submit_button = gr.Button("Explore books", variant="primary")

    summary_output = gr.Markdown()
    cards_output = gr.HTML()

    submit_button.click(
        # Gradio calls this function with the widget values in the same order
        # as the `inputs=[...]` list, then sends the returned values to `outputs=[...]`.
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown, min_rating, sort_dropdown, max_results],
        outputs=[summary_output, cards_output],
    )


if __name__ == "__main__":
    dashboard.launch()
