"""
Check one-off names against English Wikipedia.

Behavior:
- If check_names.json does not exist, initialize it from name_one_off.json
  as [{"name": ..., "status": "unknown"}, ...], shuffle order, and save.
- Then process entries with status == "unknown" one at a time:
  - query English Wikipedia search API
  - set status to "exists" or "not exists"
  - write updated check_names.json
  - print name/status and fraction of checked names that exist
- Query rate is capped at 1 request/minute.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent
ONE_OFF_PATH = BASE_DIR / "name_one_off.json"
CHECK_PATH = BASE_DIR / "check_names.json"
LEGACY_CHECK_PATH = BASE_DIR / "checked_names.json"

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "dma-tests-name-checker/1.0 (research script)"
REQUEST_INTERVAL_SECONDS = 10.0


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_checked_names(path: Path, checked_names: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(checked_names, f, ensure_ascii=False, indent=2)


def initialize_checked_names() -> list[dict[str, str]]:
    if not ONE_OFF_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {ONE_OFF_PATH}")

    names_obj = load_json(ONE_OFF_PATH)
    if not isinstance(names_obj, list):
        raise ValueError(f"{ONE_OFF_PATH} must contain a JSON list of names.")

    checked_names: list[dict[str, str]] = []
    for value in names_obj:
        name = str(value).strip()
        if not name:
            continue
        checked_names.append({"name": name, "status": "unknown"})

    random.shuffle(checked_names)
    save_checked_names(CHECK_PATH, checked_names)
    return checked_names


def load_or_initialize_checked_names() -> list[dict[str, str]]:
    source_path = CHECK_PATH
    if not source_path.exists() and LEGACY_CHECK_PATH.exists():
        source_path = LEGACY_CHECK_PATH

    if source_path.exists():
        obj = load_json(source_path)
        if not isinstance(obj, list):
            raise ValueError(f"{source_path} must contain a JSON list.")
        checked_names: list[dict[str, str]] = []
        for row in obj:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            status = str(row.get("status", "unknown")).strip() or "unknown"
            if not name:
                continue
            if status not in {"unknown", "exists", "not exists"}:
                status = "unknown"
            checked_names.append({"name": name, "status": status})
        if source_path != CHECK_PATH:
            save_checked_names(CHECK_PATH, checked_names)
        return checked_names

    return initialize_checked_names()


def wikipedia_has_article(name: str) -> bool:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": name,
        "srlimit": 1,
        "format": "json",
        "utf8": 1,
    }
    url = f"{WIKIPEDIA_API_URL}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    search_results = payload.get("query", {}).get("search", [])
    return bool(search_results)


def print_exists_fraction(checked_names: list[dict[str, str]]) -> None:
    checked_total = sum(1 for row in checked_names if row["status"] in {"exists", "not exists"})
    exists_total = sum(1 for row in checked_names if row["status"] == "exists")
    fraction = (exists_total / checked_total) if checked_total else 0.0
    print(f"exists fraction among checked names: {exists_total}/{checked_total} ({fraction:.4f})")


def main() -> None:
    checked_names = load_or_initialize_checked_names()

    last_query_time = 0.0
    for row in checked_names:
        if row["status"] != "unknown":
            continue

        elapsed = time.time() - last_query_time
        if elapsed < REQUEST_INTERVAL_SECONDS:
            time.sleep(REQUEST_INTERVAL_SECONDS - elapsed)

        name = row["name"]
        has_article = wikipedia_has_article(name)
        row["status"] = "exists" if has_article else "not exists"
        last_query_time = time.time()

        save_checked_names(CHECK_PATH, checked_names)
        print(f"{name}\t{row['status']}")
        print_exists_fraction(checked_names)

    save_checked_names(CHECK_PATH, checked_names)
    print_exists_fraction(checked_names)


if __name__ == "__main__":
    main()
