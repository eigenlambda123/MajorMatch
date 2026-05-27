import csv
from pathlib import Path
from typing import Dict, List

from course_index import (
    CourseIndexError,
    get_projection_points,
    list_courses,
    load_course_corpus,
    rebuild_course_index,
    search_courses,
)


def load_courses(path: str = "data/courses.csv") -> List[Dict]:
    return load_course_corpus(Path(path))


def semantic_search(query: str, courses: List[Dict] | None = None, top_k: int = 5) -> List[Dict]:
    """Perform a semantic search over the course corpus."""
    # Normalize inputs
    try:
        q = "" if query is None else str(query)
    except Exception:
        q = ""
    try:
        k = max(1, min(int(top_k), 5))
    except Exception:
        k = 5

    raw = search_courses(q, top_k=k)

    results: List[Dict] = []
    for item in raw:
        # item expected to have title, description, embedding, id and optionally score
        score = float(item.get("score", 0.0)) if item.get("score") is not None else 0.0
        # normalize cosine [-1,1] to [0,1]
        normalized = (score + 1.0) / 2.0
        out = {
            "id": item.get("id"),
            "title": item.get("title"),
            "description": item.get("description"),
            "score": float(score),
            "score_normalized": float(normalized),
            "source": "db",
            # include projection coords if available
            "pca_x": item.get("pca_x"),
            "pca_y": item.get("pca_y"),
            "umap_x": item.get("umap_x"),
            "umap_y": item.get("umap_y"),
            "tsne_x": item.get("tsne_x"),
            "tsne_y": item.get("tsne_y"),
        }
        results.append(out)

    return results


def get_course_projection_points(method: str = "pca"):
    """Get course projection points for the specified method (pca, umap, tsne)."""
    return get_projection_points(method)


def rebuild_index(path: str = "data/courses.csv") -> int:
    """Rebuild the course index from the specified CSV file."""
    return rebuild_course_index(Path(path))

