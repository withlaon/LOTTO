"""Static file server — serves the web/ directory on $PORT (default 8080)."""
import os
from pathlib import Path
from flask import Flask, send_from_directory

BASE_DIR = Path(__file__).resolve().parent / "web"
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")


@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(str(BASE_DIR), path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
