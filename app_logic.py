from pathlib import Path
from typing import Dict, List, Sequence

from api.jobs import get_career_context
from api.predict import predict_track
from api.search import semantic_search


BASE_DIR = Path(__file__).resolve().parent
# Point to the data directory — support multiple CSV files under `data/`.
DATA_DIR = BASE_DIR / "data"
# Collect CSV files if present; callers can use `DATA_DIR` or `COURSE_CSVS`.
COURSE_CSVS = sorted([p for p in DATA_DIR.glob("*.csv")]) if DATA_DIR.exists() else []

def recommend_track(features: dict | Sequence[str]) -> Dict[str, object]:
    """
    Recommend a track based on either a features dict (deprecated profile-like dict)
    or a sequence of selected feature names. For compatibility, forward to
    `predict_track()` which accepts either a dict or a sequence.
    """
    prediction = predict_track(features)
    if isinstance(prediction, dict):
        result = {
            "track": prediction.get("label"),
            "confidence": prediction.get("confidence"),
            "category": prediction.get("category"),
            "source": prediction.get("source"),
        }
        if "top_predictions" in prediction:
            result["top_predictions"] = prediction.get("top_predictions")
        return result

    track, confidence = prediction
    return {"track": track, "confidence": confidence}


def suggest_career_context(track: str, location: str = "Philippines") -> List[Dict[str, str]]:
    return get_career_context(track, location=location)


def should_open_prediction_tool(message: str) -> bool:
    """Return True when the user is clearly asking for a track prediction."""
    lowered = (message or "").lower()
    prediction_keywords = (
        "recommend",
        "recommendation",
        "career track",
        "predict",
        "prediction",
        "what should i study",
        "what career",
    )
    return any(keyword in lowered for keyword in prediction_keywords)


def summarize_matches(courses: Sequence[Dict[str, str]]) -> str:
    if not courses:
        return "No matches found yet. Try a broader search phrase."
    titles = [course.get("title", "") for course in courses[:5] if course.get("title")]
    if not titles:
        return "No course titles available in the current results."
    if len(titles) == 1:
        return f"Top match: {titles[0]}"
    return f"Top matches: {', '.join(titles)}"
