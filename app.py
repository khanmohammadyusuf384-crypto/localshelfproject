from pathlib import Path
import os
import threading

from prepare_deploy import main as prepare_deploy


BASE_DIR = Path(__file__).resolve().parent

if not (BASE_DIR / "chroma_db" / "chroma.sqlite3").exists():
    prepare_deploy()

from localshelf_explorer import CSS, dashboard


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST") or ("0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
    dashboard.launch(server_name=host, server_port=port, css=CSS, prevent_thread_lock=True)
    threading.Event().wait()
