from __future__ import annotations

import csv
import os
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from sqlalchemy import Float, Integer, String, Text, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from sqlalchemy.dialects.postgresql import ARRAY


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_COURSE_CSV = BASE_DIR / "data" / "courses.csv"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
DATABASE_URL = os.getenv("DATABASE_URL")

# We always store embeddings as a Postgres float array. This avoids creating
# tables that require the server-side `vector` type at table-create time,
# which would fail if the extension isn't installed. If the `vector`
# extension is available, we could later migrate to it, but for now the
# ARRAY(Float) approach is most portable across developer environments.


class CourseIndexError(RuntimeError):
    pass


class Base(DeclarativeBase):
    pass


class Course(Base):
    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    # Store embeddings as a Postgres float array to maximize compatibility.
    embedding: Mapped[List[float]] = mapped_column(ARRAY(Float), nullable=False)
    pca_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pca_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    umap_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    umap_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tsne_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tsne_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


@dataclass(frozen=True)
class CoursePoint:
    id: int
    title: str
    description: str
    x: float
    y: float


def get_database_url() -> str:
    if DATABASE_URL:
        return DATABASE_URL

    # Fall back to a sensible local default for developer convenience.
    # Use the common local Postgres defaults the user provided: postgres/postgres
    default_url = "postgresql+psycopg2://postgres:postgres@localhost:5432/semantic_search"
    return default_url


def get_engine():
    return create_engine(get_database_url(), future=True)


def get_session_factory():
    engine = get_engine()
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def ensure_database() -> None:
    engine = get_engine()
    with engine.begin() as connection:
        # Try to create the vector extension if available on the server. If
        # it is not, continue — we store embeddings as float[] and compute
        # similarities in Python.
        try:
            connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception:
            pass
    Base.metadata.create_all(engine)



def load_course_corpus(csv_path: Path | str = DEFAULT_COURSE_CSV) -> List[Dict[str, str]]:
    path = Path(csv_path)
    rows: List[Dict[str, str]] = []

    # If a directory is provided, read all CSV files inside it.
    files = []
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
    else:
        files = [path]

    for file in files:
        if not file.exists():
            continue
        with file.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                title = (row.get("title") or "").strip()
                description = (row.get("description") or "").strip()
                if title and description:
                    rows.append({"title": title, "description": description})
    return rows


def _load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def compute_embeddings(texts: Sequence[str]) -> np.ndarray:
    model = _load_embedding_model()
    embeddings = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


def _projection_matrix(embeddings: np.ndarray, method: str) -> np.ndarray:
    method = method.lower().strip()
    if embeddings.shape[0] == 1:
        return np.array([[0.0, 0.0]], dtype=np.float32)

    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
        return reducer.fit_transform(embeddings)

    if method == "umap":
        import umap

        neighbors = max(2, min(5, embeddings.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=neighbors,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(embeddings)

    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = max(2, min(5, embeddings.shape[0] - 1))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
        return reducer.fit_transform(embeddings)

    raise CourseIndexError(f"Unsupported projection method: {method}")


def rebuild_course_index(csv_path: Path | str = DEFAULT_COURSE_CSV) -> int:
    ensure_database()
    rows = load_course_corpus(csv_path)
    if not rows:
        return 0

    texts = [f"{row['title']} {row['description']}" for row in rows]
    embeddings = compute_embeddings(texts)
    projections = {
        method: _projection_matrix(embeddings, method)
        for method in ("pca", "umap", "tsne")
    }

    session_factory = get_session_factory()
    with session_factory() as session:
        session.execute(delete(Course))
        for index, row in enumerate(rows):
            course = Course(
                title=row["title"],
                description=row["description"],
                embedding=embeddings[index].tolist(),
                pca_x=float(projections["pca"][index][0]),
                pca_y=float(projections["pca"][index][1]),
                umap_x=float(projections["umap"][index][0]),
                umap_y=float(projections["umap"][index][1]),
                tsne_x=float(projections["tsne"][index][0]),
                tsne_y=float(projections["tsne"][index][1]),
            )
            session.add(course)
        session.commit()
    clear_course_caches()
    return len(rows)


@lru_cache(maxsize=1)
def _cached_course_rows() -> tuple[dict, ...]:
    return tuple(list_courses())


@lru_cache(maxsize=1)
def _cached_course_dataset() -> tuple[tuple[dict, ...], np.ndarray]:
    courses = _cached_course_rows()
    if not courses:
        return courses, np.empty((0, 0), dtype=np.float32)

    embeddings = np.asarray([course.get("embedding") for course in courses], dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = np.stack(embeddings)
    return courses, embeddings


def clear_course_caches() -> None:
    _cached_course_rows.cache_clear()
    _cached_course_dataset.cache_clear()


def _rank_courses(query: str, top_k: int) -> tuple[List[Dict[str, str]], np.ndarray, tuple[dict, ...]]:
    courses, embeddings = _cached_course_dataset()
    if not courses:
        return [], np.empty((0, 0), dtype=np.float32), courses

    query_embedding = compute_embeddings([query])[0].astype(np.float32)
    q = np.asarray(query_embedding, dtype=np.float32)

    def cosine(a, b):
        a_norm = a / (np.linalg.norm(a) + 1e-12)
        b_norm = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a_norm, b_norm))

    scores = [cosine(q, emb) for emb in embeddings]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, score in ranked:
        item = dict(courses[idx])
        item["score"] = score
        results.append(item)

    return results, q, courses


def _get_projection_columns(method: str):
    method = method.lower().strip()
    mapping = {
        "pca": (Course.pca_x, Course.pca_y),
        "umap": (Course.umap_x, Course.umap_y),
        "tsne": (Course.tsne_x, Course.tsne_y),
    }
    if method not in mapping:
        raise CourseIndexError(f"Unsupported projection method: {method}")
    return mapping[method]


def search_courses(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    if not query.strip():
        return list_courses(limit=top_k)

    ensure_database()
    results, _, _ = _rank_courses(query, top_k)
    return results


def list_courses(limit: Optional[int] = None) -> List[Dict[str, str]]:
    ensure_database()
    session_factory = get_session_factory()
    with session_factory() as session:
        stmt = select(Course)
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = session.execute(stmt).scalars().all()
        return [course_to_dict(course) for course in rows]


def get_projection_points(method: str = "pca") -> List[CoursePoint]:
    ensure_database()
    x_col, y_col = _get_projection_columns(method)
    session_factory = get_session_factory()
    with session_factory() as session:
        stmt = select(Course.id, Course.title, Course.description, x_col, y_col).order_by(Course.id)
        rows = session.execute(stmt).all()
        points: List[CoursePoint] = []
        for row in rows:
            x_val = row[3]
            y_val = row[4]
            if x_val is None or y_val is None:
                continue
            points.append(
                CoursePoint(
                    id=row[0],
                    title=row[1],
                    description=row[2],
                    x=float(x_val),
                    y=float(y_val),
                )
            )
        return points


def project_courses_with_query(query: str, method: str = "pca"):
    """Project all course embeddings together with a single query string.

    Returns (course_points, query_point) where course_points is a list of
    CoursePoint matching the order of courses returned by `list_courses()`
    and query_point is a CoursePoint with id=-1 and title="(query)".
    """
    ensure_database()
    # Load courses and their embeddings from the in-memory cache.
    courses, embeddings = _cached_course_dataset()
    if not courses:
        return [], None

    q_emb = compute_embeddings([query])[0].astype(np.float32)
    return _project_courses_with_embedding(courses, embeddings, q_emb, query, method)


def _project_courses_with_embedding(
    courses: tuple[dict, ...],
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    query_text: str,
    method: str,
):
    stacked = np.vstack([embeddings, query_embedding])
    projected = _projection_matrix(stacked, method)

    course_points: List[CoursePoint] = []
    for idx, c in enumerate(courses):
        x = float(projected[idx][0])
        y = float(projected[idx][1])
        course_points.append(CoursePoint(id=c.get("id"), title=c.get("title"), description=c.get("description"), x=x, y=y))

    qx = float(projected[-1][0])
    qy = float(projected[-1][1])
    query_point = CoursePoint(id=-1, title="(query)", description=query_text, x=qx, y=qy)
    return course_points, query_point


def search_courses_with_projection(query: str, top_k: int = 5, method: str = "pca"):
    """Return search matches plus a single projection view for the same query.

    This shares the cached course corpus and computes the query embedding once.
    """
    if not query.strip():
        results = list_courses(limit=top_k)
        return results, {
            "available": False,
            "method": method,
            "courses": [],
            "query_point": None,
            "methods": {},
        }

    ensure_database()
    results, query_embedding, courses = _rank_courses(query, top_k)
    _, embeddings = _cached_course_dataset()
    methods = {}
    for projection_method in ("pca", "umap", "tsne"):
        course_points, query_point = _project_courses_with_embedding(
            courses,
            embeddings,
            query_embedding,
            query,
            projection_method,
        )
        methods[projection_method] = {
            "available": True,
            "method": projection_method,
            "courses": [
                {
                    "id": point.id,
                    "title": point.title,
                    "description": point.description,
                    "x": point.x,
                    "y": point.y,
                }
                for point in course_points
            ],
            "query_point": None
            if query_point is None
            else {
                "id": query_point.id,
                "title": query_point.title,
                "description": query_point.description,
                "x": query_point.x,
                "y": query_point.y,
            },
        }

    selected_projection = methods.get(method) or methods.get("pca") or {
        "available": False,
        "method": method,
        "courses": [],
        "query_point": None,
    }
    projection = {
        "available": True,
        "method": selected_projection["method"],
        "courses": selected_projection["courses"],
        "query_point": selected_projection["query_point"],
        "methods": methods,
    }
    return results, projection


def course_to_dict(course: Course) -> Dict[str, str]:
    # Include the stored embedding so callers can compute similarity when the
    # server doesn't support pgvector. The embedding is stored as a list of
    # floats (Postgres float[]), so return it as-is.
    return {
        "id": int(course.id) if course.id is not None else None,
        "title": course.title,
        "description": course.description,
        "embedding": list(course.embedding) if course.embedding is not None else None,
        "pca_x": float(course.pca_x) if course.pca_x is not None else None,
        "pca_y": float(course.pca_y) if course.pca_y is not None else None,
        "umap_x": float(course.umap_x) if course.umap_x is not None else None,
        "umap_y": float(course.umap_y) if course.umap_y is not None else None,
        "tsne_x": float(course.tsne_x) if course.tsne_x is not None else None,
        "tsne_y": float(course.tsne_y) if course.tsne_y is not None else None,
    }
