from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import urllib.parse
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
LABEL_TYPES = [
    "full_name",
    "street_city_address",
    "place_name",
    "profession",
    "disease",
    "crime",
    "finance",
    "social_security_number",
    "email_address",
    "credit_card_number",
    "phone_number",
]


def tokenize(text: str) -> set[str]:
    return {tok for tok in TOKEN_RE.findall(text.lower()) if tok}


def extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""


def coerce_query_labels(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, list):
            return converted

    return []


def normalize_label_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


@dataclass
class LabelSummary:
    span_count: int
    type_counts: dict[str, int]
    type_set: set[str]
    pair_set: set[str]
    texts_by_type: dict[str, set[str]]


def build_label_summary(query_labels: list[dict[str, Any]]) -> LabelSummary:
    type_counts: Counter[str] = Counter()
    texts_by_type: dict[str, set[str]] = {label: set() for label in LABEL_TYPES}
    pair_set: set[str] = set()
    span_count = 0

    for span in query_labels:
        if not isinstance(span, dict):
            continue
        label = str(span.get("label", "")).strip()
        if label not in LABEL_TYPES:
            continue
        text = normalize_label_text(str(span.get("text", "")))
        span_count += 1
        type_counts[label] += 1
        if text:
            texts_by_type[label].add(text)
            pair_set.add(f"{label}\t{text}")

    type_set = {label for label, count in type_counts.items() if count > 0}
    return LabelSummary(
        span_count=span_count,
        type_counts=dict(type_counts),
        type_set=type_set,
        pair_set=pair_set,
        texts_by_type=texts_by_type,
    )


def build_label_pair_features(a: LabelSummary, b: LabelSummary) -> dict[str, int | float]:
    type_overlap = len(a.type_set & b.type_set)
    type_union = len(a.type_set | b.type_set)
    text_overlap = len(a.pair_set & b.pair_set)
    text_union = len(a.pair_set | b.pair_set)

    out: dict[str, int | float] = {
        "label_span_count_a": int(a.span_count),
        "label_span_count_b": int(b.span_count),
        "label_span_count_abs_diff": int(abs(a.span_count - b.span_count)),
        "has_label_a": int(a.span_count > 0),
        "has_label_b": int(b.span_count > 0),
        "both_have_label": int(a.span_count > 0 and b.span_count > 0),
        "label_type_overlap": int(type_overlap),
        "label_type_union": int(type_union),
        "label_type_jaccard": float(type_overlap / type_union) if type_union else 0.0,
        "label_text_overlap": int(text_overlap),
        "label_text_union": int(text_union),
        "label_text_jaccard": float(text_overlap / text_union) if text_union else 0.0,
    }

    for label in LABEL_TYPES:
        count_a = int(a.type_counts.get(label, 0))
        count_b = int(b.type_counts.get(label, 0))
        shared_text = len(a.texts_by_type[label] & b.texts_by_type[label])
        out[f"label_count_{label}_a"] = count_a
        out[f"label_count_{label}_b"] = count_b
        out[f"both_have_{label}"] = int(count_a > 0 and count_b > 0)
        out[f"shared_text_{label}"] = int(shared_text)

    return out
