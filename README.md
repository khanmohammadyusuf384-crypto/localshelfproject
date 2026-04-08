# LocalShelf Explorer

LocalShelf Explorer is a local-first book discovery app built around semantic search, lightweight mood filtering, and a small offline-friendly recommendation workflow. The project turns a catalog of books into a searchable shelf that can answer natural-language prompts such as:

* "a thoughtful story about forgiveness"
* "a suspenseful historical mystery"
* "a short, highly rated nonfiction read"

The project is designed to work on a personal laptop with no paid API usage. Once the local data and free Hugging Face models are prepared, the app can run without an internet connection for its core recommendation flow.

## Key features

This project is centered around a local workflow.

* It uses a local Sentence Transformers embedding model instead of paid embedding APIs.
* It persists a local Chroma vector database on disk.
* It includes a preparation script that generates the local files the app needs.
* It supports offline startup after the required model has been cached.
* It offers two discovery modes:
  * semantic search when a query is provided
  * browse mode when the query is left blank

The Gradio app is shaped as a browsing tool. It includes:

* an interface named `LocalShelf Explorer`
* shelf and mood filters
* minimum-rating filtering
* multiple sort modes
* card-style recommendations with metadata summaries

## Project layout

Core files:

* `localshelf_explorer.py`  
  Runs the local web app and renders the recommendation interface.
* `build_localshelf_catalog.py`  
  Prepares the local dataset, emotion scores, tagged descriptions, and vector store.
* `localshelf_embeddings.py`  
  Loads the local embedding model in offline-friendly mode.

Generated local assets:

* `books_cleaned.csv`
* `books_with_emotions.csv`
* `tagged_description.txt`
* `chroma_db/`
* `.hf-cache/`

Supporting files:

* `requirements.txt`
* `requirements-local.txt`
* `cover-not-found.jpg`
* notebooks for data exploration and experimentation

## How it works

1. A source book dataset is loaded or reused locally.
2. The dataset is cleaned and reduced to books with enough descriptive text.
3. A simple category mapping is added to support shelf-style filtering.
4. An emotion classifier scores each book description.
5. Tagged descriptions are written into a local text file.
6. A Chroma vector store is built using the cached local embedding model.
7. The Gradio app loads the vector store and lets you search or browse the catalog.

## Local setup

Run the commands below from this folder:

```powershell
python -m pip install -r requirements-local.txt
python build_localshelf_catalog.py
python localshelf_explorer.py
```

If the embedding model has not been cached yet, the first setup may download the free model:

* `sentence-transformers/all-MiniLM-L6-v2`

After the model has been downloaded once, it is stored in `.hf-cache/` and reused locally.

## Offline behavior

The core recommendation pipeline can run offline after setup is complete:

* local model loading
* local semantic search
* local vector database access
* local filtering and ranking
* local Gradio app on `127.0.0.1`

Current limitation:

* many book covers still come from remote thumbnail URLs stored in the dataset, so recommendation cards may appear without cover images when the internet is off

## App behavior

The dashboard supports:

* natural-language semantic search
* blank-query browsing mode
* shelf filtering by simplified categories
* mood filtering using local emotion scores
* minimum-rating filtering
* result sorting by semantic match, rating, recency, or shorter reads

Each result card shows:

* title
* author list
* short description preview
* category
* average rating
* publication year
* page count

## Notes

* `build_localshelf_catalog.py` reuses `books_cleaned.csv` if it already exists.
* The local embedding loader is configured to prefer cached files and offline startup.
* If you delete `.hf-cache/` or `chroma_db/`, those resources will need to be rebuilt.

## Future improvements

Some natural next steps for this project:

* download cover images locally for a more complete offline experience
* add bookmarking or reading-list support
* add author or publication-year filters
* surface similarity scores in the UI
* package the app into a cleaner desktop-friendly launcher
