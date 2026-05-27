from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _load_env_file() -> None:
    """Load a local .env file from the repo root if present."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    env_path = os.path.join(repo_root, ".env")
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


_load_env_file()

ADZUNA_BASE_URL = os.getenv("ADZUNA_BASE_URL", "https://api.adzuna.com/v1/api/jobs")
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "")
ADZUNA_COUNTRY = os.getenv("ADZUNA_COUNTRY", "us")
ADZUNA_CURRENCY = os.getenv("ADZUNA_CURRENCY", "USD")


@dataclass(frozen=True)
class CareerContext:
    """Structured career context data for a given track and location."""
    track: str
    location: str
    source: str
    available: bool
    job_count: Optional[int] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    salary_currency: str = ADZUNA_CURRENCY
    top_job_titles: Optional[List[str]] = None
    top_companies: Optional[List[str]] = None
    note: Optional[str] = None
    query_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track": self.track,
            "location": self.location,
            "source": self.source,
            "available": self.available,
            "job_count": self.job_count,
            "salary_min": self.salary_min,
            "salary_max": self.salary_max,
            "salary_currency": self.salary_currency,
            "top_job_titles": self.top_job_titles or [],
            "top_companies": self.top_companies or [],
            "note": self.note,
            "query_url": self.query_url,
        }


def build_job_query(track: str) -> str:
    """Normalize the input track string and build a search query for the jobs API."""
    normalized = (track or "").strip().lower()

    if any(keyword in normalized for keyword in ("software", "computer", "information technology", "engineering", "electronics", "electrical", "mechanical", "civil", "bca", "cs")):
        return "software engineer OR developer OR programmer"

    if any(keyword in normalized for keyword in ("data", "science", "math", "statistics", "analytics", "machine learning", "economics", "accounting", "research")):
        return "data scientist OR machine learning OR analytics"

    if any(keyword in normalized for keyword in ("design", "art", "visual", "architecture", "photography", "fashion", "journalism", "graphics")):
        return "product designer OR ux designer OR ui designer"

    track_map = {
        "Software Engineer": "software engineer OR developer OR programmer",
        "Data Scientist": "data scientist OR machine learning OR analytics",
        "Product Designer": "product designer OR ux designer OR ui designer",
    }
    return track_map.get(track, normalized)


def _has_credentials() -> bool:
    return bool(ADZUNA_APP_ID and ADZUNA_APP_KEY)


def _build_query_url(track: str, location: str, results_per_page: int = 10) -> str:
    country = (ADZUNA_COUNTRY or "us").strip().lower()
    query = build_job_query(track)
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": results_per_page,
        "what": query,
        "content-type": "application/json",
    }
    if location.strip():
        params["where"] = location.strip()
    return f"{ADZUNA_BASE_URL}/{country}/search/1?{urlencode(params)}"


def _fetch_json(url: str) -> Dict[str, Any]:
    request = Request(url, headers={"User-Agent": "MajorMatch/1.0"})
    with urlopen(request, timeout=20) as response:
        payload = response.read().decode("utf-8")
        return json.loads(payload)


def _extract_salary_range(results: List[Dict[str, Any]]) -> tuple[Optional[int], Optional[int]]:
    salary_mins = [int(item["salary_min"]) for item in results if item.get("salary_min") is not None]
    salary_maxs = [int(item["salary_max"]) for item in results if item.get("salary_max") is not None]

    salary_min = min(salary_mins) if salary_mins else None
    salary_max = max(salary_maxs) if salary_maxs else None
    return salary_min, salary_max


def _extract_titles(results: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    titles: List[str] = []
    for item in results:
        title = str(item.get("title") or "").strip()
        if title and title not in titles:
            titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def _extract_companies(results: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    companies: List[str] = []
    for item in results:
        company = item.get("company") or {}
        name = str(company.get("display_name") or "").strip()
        if name and name not in companies:
            companies.append(name)
        if len(companies) >= limit:
            break
    return companies


def get_career_context(track: str, location: str = "United States", results_per_page: int = 10) -> CareerContext:
    """Fetch career context for a given track and location using the Adzuna API."""
    if not track.strip():
        return CareerContext(
            track=track,
            location=location,
            source="Adzuna",
            available=False,
            note="No career track provided.",
        )

    if not _has_credentials():
        return CareerContext(
            track=track,
            location=location,
            source="Adzuna",
            available=False,
            note="Set ADZUNA_APP_ID and ADZUNA_APP_KEY to enable live job-market data.",
        )

    query_url = _build_query_url(track, location, results_per_page=results_per_page)
    try:
        payload = _fetch_json(query_url)
        results = payload.get("results", []) or []
        if not isinstance(results, list):
            results = []
        job_count = int(payload.get("count") or len(results))
        salary_min, salary_max = _extract_salary_range(results)
        return CareerContext(
            track=track,
            location=location,
            source="Adzuna",
            available=True,
            job_count=job_count,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=ADZUNA_CURRENCY,
            top_job_titles=_extract_titles(results),
            top_companies=_extract_companies(results),
            note=None,
            query_url=query_url,
        )
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as error:
        return CareerContext(
            track=track,
            location=location,
            source="Adzuna",
            available=False,
            note=f"Could not fetch job-market data: {error}",
            query_url=query_url,
        )
