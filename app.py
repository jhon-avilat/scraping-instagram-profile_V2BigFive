"""Dashboard that scrapes a profile, shows likes and comments, and estimates its Big Five traits."""

from __future__ import annotations

import json
import os
import re
import threading
import webbrowser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

import analyzer
import scraper

DATA_DIR = Path(__file__).parent / "data"
URL = "http://127.0.0.1:5000"
USERNAME_PATTERN = re.compile(r"[A-Za-z0-9._]{1,30}")
MAX_POSTS = 200
MAX_DETAILS = 30
STOPWORDS = {
    "photo", "image", "may", "with", "that", "this", "from", "and", "the", "for", "text",
    "are", "can", "has", "have", "was", "who", "his", "her", "their", "there", "into",
    "foto", "imagen", "puede", "ser", "una", "con", "los", "las", "del", "por", "que",
    "para", "texto",
}



class Job:
    """State of one background task, safe to read and update from several threads."""

    def __init__(self, **defaults):
        self._lock = threading.Lock()
        self._defaults = {"state": "idle", "error": None, **defaults}
        self._state = dict(self._defaults)

    def start(self, **fields) -> bool:
        """Reset the job and mark it as running; return False if it is already running."""
        with self._lock:
            if self._state["state"] == "running":
                return False
            self._state = {**self._defaults, **fields, "state": "running"}
            return True

    def update(self, **fields) -> None:
        """Merge `fields` into the current state."""
        with self._lock:
            self._state.update(fields)

    def snapshot(self) -> dict:
        """Return a copy of the current state."""
        with self._lock:
            return dict(self._state)


app = Flask(__name__)
scrape_job = Job(username=None, stage=None, progress=0, total=0, message=None)
analysis_job = Job(progress=0, total=0)


def load_json(filename: str, default):
    """Return the parsed content of a file in data/, or `default` if it does not exist yet."""
    path = DATA_DIR / filename
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def average(values: list[int]) -> int | None:
    """Return the rounded mean of `values`, or None when there is nothing to average."""
    return round(sum(values) / len(values)) if values else None


def compute_engagement(posts: list[dict]) -> dict:
    """Summarize likes and comments over the posts that were opened in detail."""
    detailed = [post for post in posts if "caption" in post]
    liked = [post for post in detailed if post.get("likes") is not None]
    top = max(liked, key=lambda post: post["likes"], default=None)
    return {
        "detailed": len(detailed),
        "avg_likes": average([post["likes"] for post in liked]),
        "avg_comments": average([post["comments_count"] for post in detailed if post.get("comments_count") is not None]),
        "comments_read": sum(len(post["comments"]) for post in detailed),
        "top_url": top["url"] if top else None,
        "top_likes": top["likes"] if top else None,
    }


def compute_stats(posts: list[dict]) -> dict:
    """Summarize posts: content mix, description coverage, frequent words and engagement."""
    lengths = [len(post["description"]) for post in posts if post["description"]]
    words = Counter(
        word
        for post in posts
        for word in re.findall(r"[^\W\d_]{4,}", (post.get("caption") or post["description"]).lower())
        if word not in STOPWORDS
    )
    return {
        "engagement": compute_engagement(posts),
        "total": len(posts),
        "types": dict(Counter(post["type"] for post in posts)),
        "described": len(lengths),
        "avg_description_length": round(sum(lengths) / len(lengths)) if lengths else 0,
        "top_words": [{"word": word, "count": count} for word, count in words.most_common(10)],
    }


def run_scrape(username: str, target: int, details: int, headless: bool) -> None:
    """Run the scraper in a background thread and publish its outcome in `scrape_job`."""
    try:
        scraper.run(
            username,
            target,
            headless,
            details,
            on_progress=lambda stage, done, total: scrape_job.update(stage=stage, progress=done, total=total),
            on_status=lambda message: scrape_job.update(message=message),
        )
        scrape_job.update(state="done")
    except Exception as error:  # any failure must reach the UI instead of dying in the thread
        scrape_job.update(state="error", error=str(error))


def run_analysis(runs: int) -> None:
    """Run the Gemini analysis in a background thread and publish its outcome in `analysis_job`."""
    try:
        analyzer.run(runs, on_progress=lambda done, total: analysis_job.update(progress=done, total=total))
        analysis_job.update(state="done")
    except Exception as error:  # any failure must reach the UI instead of dying in the thread
        analysis_job.update(state="error", error=str(error))


@app.get("/")
def index():
    """Serve the dashboard page."""
    return render_template("index.html")


@app.get("/api/data")
def api_data():
    """Return the saved profile, posts, statistics, analysis and self-report as JSON.

    The analysis and the self-report are returned only if they belong to the profile that
    is currently saved, so results of a previous profile are never shown under a new one.
    """
    profile = load_json("profile.json", None)
    posts = load_json("posts.json", [])
    records = {name: load_json(f"{name}.json", None) for name in ("analysis", "self_report")}
    analysis, self_report = (
        record if record and profile and record.get("username") == profile["username"] else None
        for record in records.values()
    )
    comparison = analyzer.compare(analysis, self_report["scores"]) if analysis and self_report else None
    return jsonify(
        profile=profile,
        posts=posts,
        stats=compute_stats(posts),
        analysis=analysis,
        self_report=self_report,
        comparison=comparison,
        bfi10=analyzer.BFI10_ITEMS,
    )


@app.post("/api/scrape")
def api_scrape():
    """Validate the request and start a scraping job in the background."""
    body = request.get_json(silent=True) or {}
    username = str(body.get("username", "")).strip().lstrip("@")
    if not USERNAME_PATTERN.fullmatch(username):
        return jsonify(error="Enter a valid username: letters, numbers, dots and underscores."), 400
    try:
        target = min(max(int(body.get("posts", 50)), 1), MAX_POSTS)
        details = min(max(int(body.get("details", scraper.DEFAULT_DETAILS)), 0), MAX_DETAILS)
    except (TypeError, ValueError):
        return jsonify(error="Posts and detailed posts must be numbers."), 400

    if not scrape_job.start(username=username, total=target):
        return jsonify(error="A scraping job is already running."), 409
    threading.Thread(target=run_scrape, args=(username, target, details, bool(body.get("headless"))), daemon=True).start()
    return jsonify(scrape_job.snapshot()), 202


@app.get("/api/status")
def api_status():
    """Return the state of the current scraping job."""
    return jsonify(scrape_job.snapshot())


@app.post("/api/analyze")
def api_analyze():
    """Validate the request and start the Gemini analysis in the background.

    The request must confirm that the profile owner agreed to the analysis.
    """
    body = request.get_json(silent=True) or {}
    if body.get("consent") is not True:
        return jsonify(error="Confirm that the profile owner agreed to this analysis."), 400
    if not os.environ.get("GEMINI_API_KEY"):
        return jsonify(error="GEMINI_API_KEY is not set. Add it to .env and restart the app."), 400
    try:
        runs = min(max(int(body.get("runs", analyzer.DEFAULT_RUNS)), 1), analyzer.MAX_RUNS)
    except (TypeError, ValueError):
        return jsonify(error="Runs must be a number."), 400

    if not analysis_job.start(total=runs):
        return jsonify(error="An analysis is already running."), 409
    threading.Thread(target=run_analysis, args=(runs,), daemon=True).start()
    return jsonify(analysis_job.snapshot()), 202


@app.get("/api/analysis/status")
def api_analysis_status():
    """Return the state of the current analysis job."""
    return jsonify(analysis_job.snapshot())


@app.post("/api/self-report")
def api_self_report():
    """Score a BFI-10 questionnaire answered by the profile owner and save it for comparison."""
    profile = load_json("profile.json", None)
    if not profile:
        return jsonify(error="Scrape a profile first."), 400
    body = request.get_json(silent=True) or {}
    try:
        answers = [int(answer) for answer in body.get("answers", [])]
        scores = analyzer.bfi10_scores(answers)
    except (TypeError, ValueError):
        return jsonify(error="Answer all ten statements with a number from 1 to 5."), 400

    record = {
        "username": profile["username"],
        "answers": answers,
        "scores": {trait: round(score, 2) for trait, score in scores.items()},
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "self_report.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return jsonify(record), 201


def main() -> None:
    """Load the session cookie, open the browser and start the server."""
    load_dotenv()
    threading.Timer(1.0, webbrowser.open, args=(URL,)).start()
    app.run(port=5000)


if __name__ == "__main__":
    main()
