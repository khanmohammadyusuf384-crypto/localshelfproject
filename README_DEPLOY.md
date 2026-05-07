# Deploying LocalShelf Explorer

## Hugging Face Spaces

1. Create a new Space with the Gradio SDK.
2. Push this repository to the Space.
3. Keep `app.py`, `requirements-local.txt`, `books_with_emotions.csv`, and `tagged_description.txt` in the Space repo.
4. Run `python prepare_deploy.py` during setup if `chroma_db/` is not uploaded.

The app uses `sentence-transformers/all-MiniLM-L6-v2`, which is free on Hugging Face. The first build downloads the model and creates `chroma_db/`.

## Render

Use `render.yaml`. It installs dependencies, builds the local catalog/vector store, and starts `python app.py`.

## Railway

Use `railway.json`. It follows the same install/build/start flow as Render.

## Runtime Data

Favorites are stored in `saved_books.json`. Ranking interactions are stored in `ranking_interactions.jsonl`. These are ignored by Git because they are local user state.
