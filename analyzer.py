"""Estimate Big Five personality traits from scraped Instagram posts with the Gemini API."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_RUNS = 3
MAX_RUNS = 5
MAX_ANALYZED_POSTS = 10
TEMPERATURE = 0.3
TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
CONFIDENCE_LEVELS = ("low", "medium", "high")
IMAGE_CREDIT = re.compile(r"^(?:Photos?|Video|Reel) by .+? on [^.]*\.\s*")
LIMITATIONS = (
    "Exploratory estimate made by a language model from a small sample of public posts. "
    "Public posts show how a person presents themselves, not necessarily who they are. "
    "It is not a psychological assessment and must not be used to make decisions about anyone."
)

# Rammstedt & John (2007), BFI-10. Each item follows "I see myself as someone who ...".
BFI10_ITEMS = (
    "is reserved",
    "is generally trusting",
    "tends to be lazy",
    "is relaxed, handles stress well",
    "has few artistic interests",
    "is outgoing, sociable",
    "tends to find fault with others",
    "does a thorough job",
    "gets nervous easily",
    "has an active imagination",
)
BFI10_KEY = (  # (trait, reverse-keyed) for each item above
    ("extraversion", True), ("agreeableness", False), ("conscientiousness", True),
    ("neuroticism", True), ("openness", True), ("extraversion", False),
    ("agreeableness", True), ("conscientiousness", False), ("neuroticism", False),
    ("openness", False),
)

SYSTEM_PROMPT = """You support a university exercise on computational personality assessment.
You receive anonymized, public Instagram data of one account owner: the biography without the
display name, and several posts with caption, like and comment counts, an automatic image
description and comments. Comments written by the owner are labelled "owner"; other people are
labelled user_1, user_2 and so on.

Estimate the owner's Big Five traits (openness, conscientiousness, extraversion, agreeableness,
neuroticism) on a scale from 1.0 (very low) to 5.0 (very high), where 3.0 is the population average.

Rules:
- Use only the data provided. Do not use outside knowledge and do not try to identify anyone.
- Public posts are curated self-presentation, so treat every estimate as a tentative hypothesis.
- Every score needs evidence: cite post ids such as "post_3", or "bio". Never cite ids that do not exist.
- When the evidence is thin or ambiguous, say so: use confidence "low" and keep the score close to 3.0.
- Do not stereotype from topics alone, and do not infer demographics, health conditions, sexual
  orientation, politics or religion.
- This is not a clinical or diagnostic assessment.
- Write in English, even if the posts are in another language."""

TRAIT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "score": {"type": "NUMBER", "description": "Estimate from 1.0 (very low) to 5.0 (very high); 3.0 is average."},
        "confidence": {"type": "STRING", "enum": list(CONFIDENCE_LEVELS), "description": "How much evidence supports the score."},
        "evidence": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Post ids such as post_3, or bio."},
        "rationale": {"type": "STRING", "description": "Two or three sentences explaining the score from the evidence."},
    },
    "required": ["score", "confidence", "evidence", "rationale"],
}
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        **{trait: TRAIT_SCHEMA for trait in TRAITS},
        "summary": {"type": "STRING", "description": "Three sentences summarizing the estimate and its limits."},
    },
    "required": [*TRAITS, "summary"],
}

# Human-readable labels and direction descriptions for each Big Five trait.
# Tuple format: (display_label, description_when_high, description_when_low)
TRAIT_LABELS = {
    "openness":          ("Apertura a la experiencia", "curioso, imaginativo y abierto a nuevas ideas",                "práctico, convencional y orientado a lo familiar"),
    "conscientiousness": ("Responsabilidad",            "organizado, disciplinado y orientado a metas",                 "espontáneo, flexible y menos estructurado"),
    "extraversion":      ("Extraversión",               "sociable, energético y asertivo",                              "reservado, tranquilo y que prefiere la soledad"),
    "agreeableness":     ("Amabilidad",                 "cooperativo, empático y confiado en los demás",                "competitivo, escéptico y directo"),
    "neuroticism":       ("Neuroticismo",               "propenso a la preocupación, el estrés y los cambios de humor", "emocionalmente estable y tranquilo"),
}

_CONSISTENCY_NOTE = {
    "high":   "La estimación es consistente a lo largo de múltiples ejecuciones del modelo.",
    "medium": "Existe cierta variabilidad entre ejecuciones; interpreta los valores con cautela.",
    "low":    "Alta variabilidad entre ejecuciones; la muestra es insuficiente para conclusiones firmes.",
}

log = logging.getLogger("analyzer")


class AnalysisError(RuntimeError):
    """A problem the user can act on: missing data, missing key or a failed Gemini call."""


def read_json(filename: str):
    """Load a file from data/, or raise an AnalysisError if it has not been created yet."""
    path = DATA_DIR / filename
    if not path.exists():
        raise AnalysisError("No scraped data found. Scrape a profile first.")
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload(profile: dict, posts: list[dict], limit: int = MAX_ANALYZED_POSTS) -> tuple[dict, dict[str, str]]:
    """Build the anonymized model input and the map from post ids to post URLs.

    Only text and counters leave the machine: no username, no URLs and no display name (the
    first biography line is dropped because it is usually the name). Image descriptions lose
    their 'Photo by <name> on <date>.' prefix for the same reason.
    """
    detailed = [post for post in posts if "caption" in post][:limit]
    if not detailed:
        raise AnalysisError("No posts were read in detail. Scrape again with 'Detailed posts' above 0.")

    lines = [line for line in profile.get("biography", "").splitlines() if line.strip()]
    items, sources = [], {}
    for number, post in enumerate(detailed, start=1):
        post_id = f"post_{number}"
        sources[post_id] = post["url"]
        items.append({
            "id": post_id,
            "type": post["type"],
            "posted_at": post.get("posted_at"),
            "caption": post.get("caption", ""),
            "likes": post.get("likes"),
            "comments_count": post.get("comments_count"),
            "image_description": IMAGE_CREDIT.sub("", post.get("description", "")).strip(),
            "comments": post.get("comments", []),
        })
    return {"biography": "\n".join(lines[1:]), "posts": items}, sources


def parse_run(text: str | None) -> dict:
    """Validate one Gemini answer and return it as {trait: {...}, 'summary': str}."""
    if not text:
        raise AnalysisError("Gemini returned no text; the answer may have been blocked or cut off.")
    try:
        data = json.loads(text)
        run = {"summary": str(data["summary"])}
        for trait in TRAITS:
            estimate = data[trait]
            score = estimate["score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError("score is not a number")
            confidence = str(estimate["confidence"]).lower()
            run[trait] = {
                "score": min(5.0, max(1.0, float(score))),
                "confidence": confidence if confidence in CONFIDENCE_LEVELS else "low",
                "evidence": [str(item) for item in estimate["evidence"]],
                "rationale": str(estimate["rationale"]),
            }
    except (ValueError, KeyError, TypeError) as error:
        raise AnalysisError("Gemini answered with an unexpected format; try again.") from error
    return run


def aggregate(runs: list[dict], valid_ids: set[str]) -> dict:
    """Combine several runs into mean scores, the lowest confidence and a consistency label.

    Rationale, evidence and summary come from the most representative run, the one whose
    scores are closest to the means, so the text always matches a single coherent answer.
    """
    means = {trait: statistics.fmean(run[trait]["score"] for run in runs) for trait in TRAITS}
    best = min(runs, key=lambda run: sum(abs(run[trait]["score"] - means[trait]) for trait in TRAITS))
    traits = {}
    for trait in TRAITS:
        scores = [run[trait]["score"] for run in runs]
        spread = max(scores) - min(scores)
        traits[trait] = {
            "score": round(means[trait], 2),
            "spread": round(spread, 2),
            "consistency": None if len(runs) < 2 else "high" if spread <= 0.5 else "medium" if spread <= 1.0 else "low",
            "confidence": min((run[trait]["confidence"] for run in runs), key=CONFIDENCE_LEVELS.index),
            "evidence": [item for item in dict.fromkeys(best[trait]["evidence"]) if item in valid_ids],
            "rationale": best[trait]["rationale"].strip(),
        }
    return {"traits": traits, "summary": best["summary"].strip()}


def analyze(
    payload: dict,
    sources: dict[str, str],
    generate: Callable[[str], str | None],
    runs: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Ask the model `runs` times through `generate` and aggregate the validated answers."""
    content = "Estimate the Big Five traits of the account owner from this anonymized data.\n\n" + json.dumps(
        payload, ensure_ascii=False, indent=1
    )
    results = []
    if on_progress:
        on_progress(0, runs)
    for done in range(1, runs + 1):
        results.append(parse_run(generate(content)))
        log.info("Gemini run %d/%d finished", done, runs)
        if on_progress:
            on_progress(done, runs)
    return {
        **aggregate(results, set(sources) | {"bio"}),
        "runs": runs,
        "sources": sources,
        "input": {
            "posts": len(payload["posts"]),
            "comments": sum(len(post["comments"]) for post in payload["posts"]),
            "has_biography": bool(payload["biography"]),
        },
        "limitations": LIMITATIONS,
    }


def describe_api_error(error) -> str:
    """Turn a Gemini SDK error into a message that says what to do next."""
    code, message = getattr(error, "code", None), getattr(error, "message", None) or str(error)
    if code == 429:
        return "Gemini rate limit reached. Wait a minute and try again, or check your quota in Google AI Studio."
    if code in (400, 401, 403) and "api key" in message.lower():
        return "Gemini rejected the API key. Check GEMINI_API_KEY in .env."
    if code == 404:
        return "Gemini model not found. Run `python analyzer.py --list-models` and set GEMINI_MODEL in .env."
    if code and code >= 500:
        return "Gemini is temporarily unavailable. Try again later."
    return f"Gemini error {code}: {message}"


def make_client():
    """Create a Gemini client from the GEMINI_API_KEY environment variable."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise AnalysisError("GEMINI_API_KEY is not set. Create a key in Google AI Studio and add it to .env.")
    from google import genai

    return genai.Client(api_key=key)


def make_generator(client, model: str) -> Callable[[str], str | None]:
    """Return a function that sends one prompt to Gemini and returns its JSON answer as text."""
    from google.genai import errors, types

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )

    def generate(content: str) -> str | None:
        try:
            return client.models.generate_content(model=model, contents=content, config=config).text
        except errors.APIError as error:
            raise AnalysisError(describe_api_error(error)) from error

    return generate


def list_models() -> list[str]:
    """Return the models that can generate content with the current API key."""
    from google.genai import errors

    try:
        models = list(make_client().models.list())
    except errors.APIError as error:
        raise AnalysisError(describe_api_error(error)) from error
    usable = [m for m in models if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"])]
    return sorted(m.name.removeprefix("models/") for m in usable)


def build_conclusion(traits: dict, input_info: dict) -> str:
    """Generate a human-readable personality hypothesis from aggregated Big Five scores.

    The conclusion names the dominant trait, describes its direction, adds a secondary
    trait when it deviates clearly from average, notes the lowest trait when relevant,
    reports consistency across runs, and closes with a mandatory limitations disclaimer.
    """
    sorted_traits = sorted(traits.items(), key=lambda kv: kv[1]["score"], reverse=True)
    dominant_key,  dominant  = sorted_traits[0]
    secondary_key, secondary = sorted_traits[1]
    lowest_key,    lowest    = sorted_traits[-1]

    dom_label, dom_high, dom_low   = TRAIT_LABELS[dominant_key]
    sec_label, sec_high, sec_low   = TRAIT_LABELS[secondary_key]
    low_label, _,        low_low   = TRAIT_LABELS[lowest_key]

    dom_score = dominant["score"]
    dom_conf  = dominant["confidence"]
    dom_cons  = dominant.get("consistency")

    dom_desc = dom_high if dom_score >= 3.0 else dom_low
    sec_desc = sec_high if secondary["score"] >= 3.0 else sec_low

    n_posts    = input_info.get("posts", 0)
    n_comments = input_info.get("comments", 0)
    has_bio    = input_info.get("has_biography", False)
    data_note  = (
        f"{n_posts} publicación{'es' if n_posts != 1 else ''}"
        + (f", {n_comments} comentario{'s' if n_comments != 1 else ''}" if n_comments else "")
        + (" y biografía" if has_bio else "")
    )

    conf_es = {"high": "alta", "medium": "moderada", "low": "baja"}.get(dom_conf, "baja")

    parts = [
        f"Con base en {data_note}, el rasgo más prominente de este perfil es "
        f"**{dom_label}** (puntuación {dom_score:.2f}/5, confianza {conf_es}): "
        f"la persona se muestra {dom_desc}.",
    ]

    if abs(secondary["score"] - 3.0) >= 0.25:
        parts.append(
            f"Como rasgo secundario destaca **{sec_label}** ({secondary['score']:.2f}/5), "
            f"lo que refuerza un perfil {sec_desc}."
        )

    if abs(lowest["score"] - 3.0) >= 0.3:
        direction = "bajo" if lowest["score"] < 3.0 else "elevado"
        parts.append(
            f"En contraste, **{low_label}** es el rasgo menos central "
            f"({direction}, {lowest['score']:.2f}/5)."
        )

    cons_note = _CONSISTENCY_NOTE.get(dom_cons)
    if cons_note:
        parts.append(cons_note)

    parts.append(
        "Esta hipótesis es exploratoria y se deriva de publicaciones públicas curadas; "
        "no constituye un perfil psicológico clínico ni una base para tomar decisiones sobre nadie."
    )

    return " ".join(parts)


def run(runs: int = DEFAULT_RUNS, on_progress: Callable[[int, int], None] | None = None) -> dict:
    """Analyze the saved data and write data/analysis.json; shared by the CLI and the dashboard."""
    model = os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    profile, posts = read_json("profile.json"), read_json("posts.json")
    payload, sources = build_payload(profile, posts)
    result = analyze(payload, sources, make_generator(make_client(), model), runs, on_progress)
    result.update(
        username=profile["username"],
        model=model,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        conclusion=build_conclusion(result["traits"], result["input"]),
    )
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def bfi10_scores(answers: list[int]) -> dict[str, float]:
    """Score the BFI-10 questionnaire: ten answers from 1 to 5, reverse-keyed items flipped."""
    if len(answers) != len(BFI10_KEY) or any(answer not in (1, 2, 3, 4, 5) for answer in answers):
        raise ValueError("The BFI-10 needs ten answers between 1 and 5.")
    values: dict[str, list[int]] = {trait: [] for trait in TRAITS}
    for answer, (trait, reverse) in zip(answers, BFI10_KEY):
        values[trait].append(6 - answer if reverse else answer)
    return {trait: statistics.fmean(items) for trait, items in values.items()}


def compare(analysis: dict, self_scores: dict[str, float]) -> dict:
    """Compare Gemini's scores with a self-report and return the gap per trait and on average."""
    rows = {
        trait: {
            "ai": analysis["traits"][trait]["score"],
            "self": round(self_scores[trait], 2),
            "difference": round(analysis["traits"][trait]["score"] - self_scores[trait], 2),
        }
        for trait in TRAITS
    }
    gap = statistics.fmean(abs(row["difference"]) for row in rows.values())
    return {"traits": rows, "mean_absolute_difference": round(gap, 2)}


def main() -> None:
    """Command-line entry point: analyze the saved data or list the available models."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-r", "--runs", type=int, default=DEFAULT_RUNS, help="times to ask the model (default: 3)")
    parser.add_argument("--list-models", action="store_true", help="list the models your API key can use")
    args = parser.parse_args()

    try:
        if args.list_models:
            print("\n".join(list_models()))
            return
        result = run(min(max(args.runs, 1), MAX_RUNS))
    except AnalysisError as error:
        sys.exit(f"error: {error}")
    for trait, info in result["traits"].items():
        print(f"{trait:<18} {info['score']:.2f}  confidence={info['confidence']}  consistency={info['consistency']}")
    log.info("Saved %s", DATA_DIR / "analysis.json")


if __name__ == "__main__":
    main()
