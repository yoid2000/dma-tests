"""
Distributed labeling utility for AOL query data.

Modes:
- make_distinct: build distinct_queries.parquet from raw.parquet Query values
- <i>: label chunk i (0-based) of distinct queries and write label_work/i.parquet
- create: combine label_work/*.parquet with raw.parquet into labeled.parquet
- analyze: read labeled.parquet and print label statistics (includes full_name frequency)
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
from typing import Any

try:
    from email_validator import EmailNotValidError, validate_email
except ModuleNotFoundError:
    EmailNotValidError = ValueError
    validate_email = None
import phonenumbers
import pandas as pd
import probablepeople
from stdnum import luhn
import usaddress
from gliner import GLiNER


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_PATH = BASE_DIR / "raw.parquet"
DEFAULT_OUTPUT_PATH = BASE_DIR / "labeled.parquet"
DEFAULT_DISTINCT_PATH = BASE_DIR / "distinct_queries.parquet"
DEFAULT_LABEL_WORK_DIR = BASE_DIR / "label_work"
DEFAULT_SAMPLE_OUTPUT_PATH = BASE_DIR / "samples.parquet"
DEFAULT_NUM_CHUNKS = 200

TARGET_LABELS = [
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

LOCATION_LABELS = {"street_city_address", "place_name"}
STREET_COMPONENT_KEYS = {
    "AddressNumber",
    "StreetName",
    "StreetNamePreDirectional",
    "StreetNamePreModifier",
    "StreetNamePreType",
    "StreetNamePostDirectional",
    "StreetNamePostModifier",
    "StreetNamePostType",
}
FULL_NAME_LABEL = "full_name"
EMAIL_LABEL = "email_address"
CREDIT_CARD_LABEL = "credit_card_number"
PHONE_LABEL = "phone_number"


def coerce_query_labels(value: Any) -> list[dict[str, Any]]:
    """Normalize parquet-loaded label containers to plain Python lists."""
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


def format_label_type_value_pairs(query_labels: list[dict[str, Any]]) -> list[str]:
    """Render labels as `type: value` pairs for compact human-readable output."""
    pairs: list[str] = []
    for span in query_labels:
        if not isinstance(span, dict):
            continue
        label_type = str(span.get("label", "")).strip()
        label_value = str(span.get("text", "")).strip()
        if not label_type:
            continue
        pairs.append(f"{label_type}: {label_value!r}")
    return pairs


def redact_full_name_in_query(query_text: str, query_labels: list[dict[str, Any]]) -> str:
    """Replace full_name spans in query text with the literal 'full_name'."""
    redacted = query_text
    spans: list[tuple[int, int]] = []

    for span in query_labels:
        if not isinstance(span, dict):
            continue
        if str(span.get("label", "")).strip() != "full_name":
            continue
        start_obj = span.get("start")
        end_obj = span.get("end")
        if isinstance(start_obj, int) and isinstance(end_obj, int):
            start = start_obj
            end = end_obj
            if 0 <= start < end <= len(query_text):
                spans.append((start, end))

    if spans:
        # Apply from right to left so earlier indices remain valid.
        for start, end in sorted(spans, reverse=True):
            redacted = redacted[:start] + "full_name" + redacted[end:]
        return redacted

    # Fallback if start/end are unavailable.
    for span in query_labels:
        if not isinstance(span, dict):
            continue
        if str(span.get("label", "")).strip() != "full_name":
            continue
        name_text = str(span.get("text", "")).strip()
        if name_text:
            redacted = redacted.replace(name_text, "full_name")
    return redacted


def dedupe_examples_by_query(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep first occurrence of each Query in order."""
    unique_examples: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for example in examples:
        query_text = str(example.get("Query", ""))
        if query_text in seen_queries:
            continue
        seen_queries.add(query_text)
        unique_examples.append(example)
    return unique_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distributed labeler for raw.parquet. Modes: make_distinct, sample, "
            "<chunk_index>, create, analyze."
        )
    )
    parser.add_argument(
        "mode",
        help="One of: make_distinct, sample, create, analyze, or chunk index i (0-based integer).",
    )
    parser.add_argument(
        "sample_size",
        nargs="?",
        type=int,
        default=1000,
        help="Optional sample size used only with mode=sample (default: 1000).",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help=f"Input parquet path (default: {DEFAULT_RAW_PATH}).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output parquet path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--distinct-path",
        type=Path,
        default=DEFAULT_DISTINCT_PATH,
        help=f"Distinct query parquet path (default: {DEFAULT_DISTINCT_PATH}).",
    )
    parser.add_argument(
        "--label-work-dir",
        type=Path,
        default=DEFAULT_LABEL_WORK_DIR,
        help=f"Directory for chunk label outputs (default: {DEFAULT_LABEL_WORK_DIR}).",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help=f"Number of chunks for distributed labeling (default: {DEFAULT_NUM_CHUNKS}).",
    )
    parser.add_argument(
        "--model-id",
        default="gliner-community/gliner_medium-v2.5",
        help="GLiNER model id for from_pretrained().",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum GLiNER confidence score (default: 0.5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for GLiNER inference (default: 128).",
    )
    parser.add_argument(
        "--full-name-threshold",
        type=float,
        default=0.8,
        help="Minimum confidence for full_name labels after GLiNER (default: 0.8).",
    )
    parser.add_argument(
        "--sample-output-path",
        type=Path,
        default=DEFAULT_SAMPLE_OUTPUT_PATH,
        help=f"Output path for sample mode (default: {DEFAULT_SAMPLE_OUTPUT_PATH}).",
    )
    return parser.parse_args()


def usaddress_components(text: str) -> set[str]:
    """Return component tag names found by usaddress for a location-like text."""
    try:
        tagged, _ = usaddress.tag(text)
        return set(tagged.keys())
    except usaddress.RepeatedLabelError as exc:
        return {label for _, label in exc.parsed_string}
    except Exception:
        return set()


def normalize_location_label(span_text: str) -> str | None:
    """
    Normalize location labels to:
    - street_city_address when street + city are present
    - place_name when street is absent
    Returns None when street exists but city is missing.
    """
    components = usaddress_components(span_text)
    has_street = bool(components & STREET_COMPONENT_KEYS)
    has_city = "PlaceName" in components

    if has_street and has_city:
        return "street_city_address"
    if has_street and not has_city:
        return None
    return "place_name"


def is_plausible_full_name_text(text: str) -> bool:
    """
    Heuristic pre-filter for names:
    - require at least two alphabetic tokens
    - reject url/email-like and alphanumeric/identifier-like strings
    """
    stripped = text.strip()
    if not stripped:
        return False

    # Reject common non-name patterns quickly.
    if re.search(r"(https?://|www\.|@|[0-9]|[\\/._])", stripped, flags=re.IGNORECASE):
        return False

    if re.search(r"[^A-Za-z\s'\-]", stripped):
        return False

    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]*", stripped)
    return len(tokens) >= 2


def is_valid_full_name(text: str, score: float, full_name_threshold: float) -> bool:
    """Validate that a full_name prediction is likely first+last name."""
    if score < full_name_threshold:
        return False
    if not is_plausible_full_name_text(text):
        return False

    try:
        tagged, entity_type = probablepeople.tag(text)
    except probablepeople.RepeatedLabelError:
        return False
    except Exception:
        return False

    if str(entity_type) != "Person":
        return False

    keys = set(tagged.keys())
    return "GivenName" in keys and "Surname" in keys


def normalize_email_label(span_text: str) -> str | None:
    """
    Keep email labels only when they are real email-address shaped values.
    Domain-only strings (e.g., example.com) are rejected.
    """
    candidate = span_text.strip().strip(".,;:()[]{}<>\"'")
    if "@" not in candidate:
        return None

    if validate_email is None:
        # Fallback when email_validator is unavailable on a worker node.
        if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", candidate):
            return candidate.lower()
        return None

    try:
        validated = validate_email(candidate, check_deliverability=False)
        return validated.normalized
    except EmailNotValidError:
        return None


def normalize_credit_card_label(span_text: str) -> str | None:
    """
    Keep credit-card labels only when they look like real PAN values:
    - digits with optional spaces/hyphens
    - 13-19 digits
    - Luhn-valid
    """
    candidate = span_text.strip().strip(".,;:()[]{}<>\"'")
    compact = re.sub(r"[\s\-]", "", candidate)
    if not compact.isdigit():
        return None
    if len(compact) < 13 or len(compact) > 19:
        return None
    if not luhn.is_valid(compact):
        return None
    return compact


def normalize_phone_label(span_text: str) -> str | None:
    """
    Keep phone labels only when parseable and valid in phonenumbers.
    Normalize output to E.164.
    """
    candidate = span_text.strip().strip(".,;:()[]{}<>\"'")
    if re.search(r"[A-Za-z]", candidate):
        return None

    digit_count = len(re.sub(r"\D", "", candidate))
    if digit_count < 10 or digit_count > 15:
        return None

    try:
        parsed = phonenumbers.parse(candidate, "US")
    except phonenumbers.NumberParseException:
        return None

    if not phonenumbers.is_possible_number(parsed):
        return None
    if not phonenumbers.is_valid_number(parsed):
        return None

    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)


def normalize_entity(entity: dict[str, Any], full_name_threshold: float) -> dict[str, Any] | None:
    """Normalize one GLiNER entity dict and enforce location constraints."""
    label = str(entity["label"])
    text = str(entity["text"])
    score = float(entity["score"])

    if label == FULL_NAME_LABEL:
        if not is_valid_full_name(text=text, score=score, full_name_threshold=full_name_threshold):
            return None

    if label in LOCATION_LABELS:
        normalized = normalize_location_label(text)
        if normalized is None:
            return None
        label = normalized

    if label == EMAIL_LABEL:
        normalized_email = normalize_email_label(text)
        if normalized_email is None:
            return None
        text = normalized_email

    if label == CREDIT_CARD_LABEL:
        normalized_cc = normalize_credit_card_label(text)
        if normalized_cc is None:
            return None
        text = normalized_cc

    if label == PHONE_LABEL:
        normalized_phone = normalize_phone_label(text)
        if normalized_phone is None:
            return None
        text = normalized_phone

    return {
        "label": label,
        "text": text,
        "start": int(entity["start"]),
        "end": int(entity["end"]),
        "score": score,
    }


def label_queries(
    queries: list[str],
    model: GLiNER,
    threshold: float,
    batch_size: int,
    full_name_threshold: float,
) -> list[list[dict[str, Any]]]:
    """Label all distinct queries and return mapping query -> list of span labels."""
    all_labels: list[list[dict[str, Any]]] = []

    total = len(queries)
    for offset in range(0, total, batch_size):
        batch = queries[offset : offset + batch_size]
        batch_entities = predict_entities_batch(
            model=model,
            texts=batch,
            labels=TARGET_LABELS,
            threshold=threshold,
        )

        for entities in batch_entities:
            normalized_entities: list[dict[str, Any]] = []
            for entity in entities:
                normalized = normalize_entity(entity, full_name_threshold=full_name_threshold)
                if normalized is not None:
                    normalized_entities.append(normalized)
            all_labels.append(normalized_entities)

        processed = min(offset + batch_size, total)
        if processed % 100_000 == 0 or processed == total:
            print(f"Labeled {processed:,}/{total:,} distinct queries")

    return all_labels


def predict_entities_batch(
    model: GLiNER, texts: list[str], labels: list[str], threshold: float
) -> list[list[dict[str, Any]]]:
    """
    Predict entities for a batch across GLiNER API variants.

    Supported in priority order:
    - model.inference(...)
    - model.run(...)
    - model.batch_predict_entities(...)
    - repeated model.predict_entities(...)
    """
    if hasattr(model, "inference"):
        try:
            return model.inference(texts, labels, threshold=threshold)
        except TypeError:
            return model.inference(texts, labels)

    if hasattr(model, "run"):
        return model.run(texts, labels=labels, threshold=threshold)

    if hasattr(model, "batch_predict_entities"):
        try:
            return model.batch_predict_entities(texts, labels, threshold=threshold)
        except TypeError:
            return model.batch_predict_entities(texts, labels)

    if hasattr(model, "predict_entities"):
        out: list[list[dict[str, Any]]] = []
        for text in texts:
            try:
                out.append(model.predict_entities(text, labels, threshold=threshold))
            except TypeError:
                out.append(model.predict_entities(text, labels))
        return out

    raise AttributeError("Unsupported GLiNER API on model instance.")


def build_distinct_queries(raw_path: Path, distinct_path: Path) -> None:
    """Write a parquet file containing one distinct Query per row."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {raw_path}")

    print(f"Loading data from: {raw_path}")
    df = pd.read_parquet(raw_path, columns=["Query"])
    query_series = df["Query"].astype("string").fillna("")
    distinct_df = pd.DataFrame({"Query": query_series.drop_duplicates().tolist()})

    distinct_path.parent.mkdir(parents=True, exist_ok=True)
    distinct_df.to_parquet(distinct_path, index=False)
    print(f"Wrote {len(distinct_df):,} distinct queries to: {distinct_path}")


def chunk_bounds(total: int, index: int, num_chunks: int) -> tuple[int, int]:
    """Return [start, end) indices for chunk index under floor-based partitioning."""
    start = (index * total) // num_chunks
    end = ((index + 1) * total) // num_chunks
    return start, end


def run_chunk_labeling(
    chunk_index: int,
    distinct_path: Path,
    label_work_dir: Path,
    num_chunks: int,
    model_id: str,
    threshold: float,
    batch_size: int,
    full_name_threshold: float,
) -> None:
    """Label a single distinct-query chunk and write label_work/<i>.parquet."""
    if chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError(f"Chunk index must be in [0, {num_chunks - 1}], got {chunk_index}.")
    if not distinct_path.exists():
        raise FileNotFoundError(
            f"Distinct query file not found: {distinct_path}. Run mode make_distinct first."
        )

    distinct_df = pd.read_parquet(distinct_path, columns=["Query"])
    distinct_queries = distinct_df["Query"].astype("string").fillna("").tolist()
    total = len(distinct_queries)
    start, end = chunk_bounds(total, chunk_index, num_chunks)
    queries = distinct_queries[start:end]

    print(
        f"Chunk {chunk_index}/{num_chunks - 1}: rows [{start:,}, {end:,}) "
        f"({len(queries):,} queries)"
    )
    print(f"Loading GLiNER model: {model_id}")
    model = GLiNER.from_pretrained(model_id)

    labels = label_queries(
        queries,
        model=model,
        threshold=threshold,
        batch_size=batch_size,
        full_name_threshold=full_name_threshold,
    )

    out_df = pd.DataFrame({"Query": queries, "QueryLabels": labels})
    label_work_dir.mkdir(parents=True, exist_ok=True)
    out_path = label_work_dir / f"{chunk_index}.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Wrote chunk labels: {out_path}")

    label_counter: Counter[str] = Counter()
    rows_with_labels = 0
    for row_labels in labels:
        if row_labels:
            rows_with_labels += 1
        for span in row_labels:
            label_counter[str(span["label"])] += 1

    print(f"Chunk queries: {len(queries):,}")
    print(f"Chunk rows with at least one label: {rows_with_labels:,}")
    print("Chunk span counts by label:")
    for label in TARGET_LABELS:
        print(f"  {label}: {label_counter[label]:,}")

    print("\nChunk score extremes by label:")
    print_sample_extremes(out_df)


def create_labeled_parquet(
    raw_path: Path,
    label_work_dir: Path,
    output_path: Path,
    num_chunks: int,
) -> None:
    """Join all label chunk files to raw.parquet and write labeled.parquet."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {raw_path}")
    if not label_work_dir.exists():
        raise FileNotFoundError(f"Label work directory not found: {label_work_dir}")

    expected_files = [label_work_dir / f"{i}.parquet" for i in range(num_chunks)]
    available_files = [path for path in expected_files if path.exists()]
    missing_files = [str(path) for path in expected_files if not path.exists()]
    if not available_files:
        raise FileNotFoundError(
            f"No chunk files found in {label_work_dir} among indices 0..{num_chunks - 1}."
        )
    if missing_files:
        preview = ", ".join(missing_files[:10])
        suffix = "..." if len(missing_files) > 10 else ""
        print(
            f"Using {len(available_files)} chunk files; missing {len(missing_files)}: "
            f"{preview}{suffix}"
        )

    label_frames = [
        pd.read_parquet(path, columns=["Query", "QueryLabels"]) for path in available_files
    ]
    labels_df = pd.concat(label_frames, ignore_index=True)
    labels_df = labels_df.drop_duplicates(subset=["Query"], keep="last")

    print(f"Loading raw data from: {raw_path}")
    raw_df = pd.read_parquet(raw_path)
    labeled_df = raw_df.merge(labels_df, on="Query", how="left")
    labeled_df["QueryLabels"] = labeled_df["QueryLabels"].apply(coerce_query_labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_parquet(output_path, index=False)
    print(f"Wrote merged labeled parquet: {output_path}")
    print(f"Rows: {len(labeled_df):,}")


def print_sample_extremes(sample_df: pd.DataFrame) -> None:
    """Print highest and lowest 10 scores per label from sample labels."""
    span_rows: list[dict[str, Any]] = []
    for _, row in sample_df.iterrows():
        query = str(row["Query"])
        labels = row["QueryLabels"]
        if not isinstance(labels, list):
            continue
        for span in labels:
            if not isinstance(span, dict):
                continue
            span_rows.append(
                {
                    "label": str(span.get("label", "")),
                    "score": float(span.get("score", 0.0)),
                    "query": query,
                    "text": str(span.get("text", "")),
                    "start": int(span.get("start", -1)),
                    "end": int(span.get("end", -1)),
                }
            )

    if not span_rows:
        print("No labeled spans found in sample output.")
        return

    spans_df = pd.DataFrame(span_rows)
    for label in TARGET_LABELS:
        label_df = spans_df.loc[spans_df["label"] == label].copy()
        print(f"\nLabel: {label}")
        if label_df.empty:
            print("  No spans found.")
            continue

        high = label_df.sort_values("score", ascending=False).head(10)
        low = label_df.sort_values("score", ascending=True).head(10)

        print("  Highest 10:")
        for _, r in high.iterrows():
            print(
                f"    score={r['score']:.4f} query={r['query']!r} "
                f"span={r['text']!r} [{r['start']},{r['end']})"
            )

        print("  Lowest 10:")
        for _, r in low.iterrows():
            print(
                f"    score={r['score']:.4f} query={r['query']!r} "
                f"span={r['text']!r} [{r['start']},{r['end']})"
            )


def run_sample_labeling(
    distinct_path: Path,
    sample_output_path: Path,
    model_id: str,
    threshold: float,
    batch_size: int,
    full_name_threshold: float,
    sample_size: int,
) -> None:
    """Label a random sample of distinct queries and write samples.parquet."""
    if not distinct_path.exists():
        raise FileNotFoundError(
            f"Distinct query file not found: {distinct_path}. Run mode make_distinct first."
        )
    if sample_size <= 0:
        raise ValueError(f"sample_size must be > 0, got {sample_size}.")

    distinct_df = pd.read_parquet(distinct_path, columns=["Query"])
    query_series = distinct_df["Query"].astype("string").fillna("")
    sample_n = min(sample_size, len(query_series))
    queries = query_series.sample(n=sample_n, replace=False).tolist()

    print(f"Sample mode: randomly labeling {len(queries):,} distinct queries")
    print(f"Loading GLiNER model: {model_id}")
    model = GLiNER.from_pretrained(model_id)

    labels = label_queries(
        queries,
        model=model,
        threshold=threshold,
        batch_size=batch_size,
        full_name_threshold=full_name_threshold,
    )

    sample_df = pd.DataFrame({"Query": queries, "QueryLabels": labels})
    sample_output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_parquet(sample_output_path, index=False)
    print(f"Wrote sample labels: {sample_output_path}")

    print_sample_extremes(sample_df)


def run_analyze_labels(output_path: Path) -> None:
    """Read labeled.parquet and print descriptive label statistics."""
    if not output_path.exists():
        raise FileNotFoundError(
            f"Labeled parquet not found: {output_path}. Run mode create first."
        )

    print(f"Loading labeled data from: {output_path}")
    df = pd.read_parquet(output_path, columns=["AnonID", "Query", "QueryLabels"])

    total_rows = len(df)
    if total_rows == 0:
        print("No rows found in labeled parquet.")
        return

    labels_per_row = Counter()
    label_span_counts = Counter()
    label_row_counts = Counter()
    label_text_counts: dict[str, Counter[str]] = {}
    label_score_stats: dict[str, dict[str, float]] = {}

    rows_with_any_label = 0
    total_spans = 0

    for value in df["QueryLabels"]:
        row_labels = coerce_query_labels(value)
        label_count = len(row_labels)
        labels_per_row[label_count] += 1
        if label_count > 0:
            rows_with_any_label += 1
        row_label_set: set[str] = set()

        for span in row_labels:
            if not isinstance(span, dict):
                continue

            label = str(span.get("label", ""))
            if not label:
                continue

            row_label_set.add(label)
            total_spans += 1
            label_span_counts[label] += 1

            text = str(span.get("text", "")).strip()
            if label not in label_text_counts:
                label_text_counts[label] = Counter()
            if text:
                label_text_counts[label][text] += 1

            score_obj = span.get("score", None)
            if score_obj is not None:
                try:
                    score = float(score_obj)
                except (TypeError, ValueError):
                    score = None
                if score is not None:
                    stats = label_score_stats.setdefault(
                        label,
                        {"count": 0.0, "sum": 0.0, "min": score, "max": score},
                    )
                    stats["count"] += 1.0
                    stats["sum"] += score
                    if score < stats["min"]:
                        stats["min"] = score
                    if score > stats["max"]:
                        stats["max"] = score

        for label in row_label_set:
            label_row_counts[label] += 1

    pct_any = (rows_with_any_label / total_rows) * 100.0
    print("\nRow-level stats")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Rows with >=1 label: {rows_with_any_label:,} ({pct_any:.2f}%)")
    print(f"  Rows with 0 labels: {labels_per_row[0]:,}")
    print(f"  Total labeled spans: {total_spans:,}")
    print(f"  Average labels per row: {total_spans / total_rows:.4f}")

    print("\nDistribution: labels per row")
    for label_count in sorted(labels_per_row.keys()):
        row_count = labels_per_row[label_count]
        pct = (row_count / total_rows) * 100.0
        print(f"  {label_count}: {row_count:,} rows ({pct:.2f}%)")

    print("\nSpan counts by label")
    if not label_span_counts:
        print("  No labels found.")
        return

    for label, count in sorted(label_span_counts.items(), key=lambda item: (-item[1], item[0])):
        pct = (count / total_spans) * 100.0 if total_spans else 0.0
        print(f"  {label}: {count:,} spans ({pct:.2f}% of spans)")

    print("\nRow counts by label")
    for label, count in sorted(label_row_counts.items(), key=lambda item: (-item[1], item[0])):
        pct = (count / total_rows) * 100.0 if total_rows else 0.0
        print(f"  {label}: {count:,} rows ({pct:.2f}% of rows)")

    print("\nDistinct values by label")
    for label, _ in sorted(label_span_counts.items(), key=lambda item: (-item[1], item[0])):
        text_counter = label_text_counts.get(label, Counter())
        distinct_count = len(text_counter)
        print(f"  {label}: {distinct_count:,} distinct values")

    print("\nTop values by label (up to 10)")
    for label, _ in sorted(label_span_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {label}:")
        text_counter = label_text_counts.get(label, Counter())
        if not text_counter:
            print("    (no non-empty text values)")
            continue
        for text, count in text_counter.most_common(10):
            print(f"    {count:,} x {text!r}")

    print("\nScore summary by label")
    for label, _ in sorted(label_span_counts.items(), key=lambda item: (-item[1], item[0])):
        stats = label_score_stats.get(label)
        if not stats or stats["count"] <= 0:
            print(f"  {label}: no score values")
            continue
        mean = stats["sum"] / stats["count"]
        print(
            f"  {label}: count={int(stats['count']):,} "
            f"mean={mean:.4f} min={stats['min']:.4f} max={stats['max']:.4f}"
        )

    print("\n" + "="*80)
    print("Full_name frequency distribution (by distinct AnonID)")
    print("="*80)
    
    # Track distinct AnonIDs per name
    name_anonids: dict[str, set[Any]] = {}
    
    for anonid, value in zip(df["AnonID"], df["QueryLabels"]):
        row_labels = coerce_query_labels(value)
        for span in row_labels:
            if not isinstance(span, dict):
                continue
            label = str(span.get("label", ""))
            if label != "full_name":
                continue
            text = str(span.get("text", "")).strip()
            if text:
                if text not in name_anonids:
                    name_anonids[text] = set()
                name_anonids[text].add(anonid)
    
    if not name_anonids:
        print("No full_name labels found.")
    else:
        # Convert to counts of distinct AnonIDs
        name_counts: dict[str, int] = {name: len(anonids) for name, anonids in name_anonids.items()}
        one_off_names = sorted(name for name, count in name_counts.items() if count == 1)
        one_off_path = output_path.with_name("name_one_off.json")
        with one_off_path.open("w", encoding="utf-8") as f:
            json.dump(one_off_names, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(one_off_names):,} one-off names to: {one_off_path}")
        
        bins = [
            (1, 1, "1"),
            (2, 2, "2"),
            (3, 3, "3"),
            (4, 4, "4"),
            (5, 5, "5"),
            (6, 7, "6-7"),
            (8, 10, "8-10"),
            (11, 15, "11-15"),
            (16, 20, "16-20"),
            (21, 50, "21-50"),
            (51, 100, "51-100"),
            (101, 999_999_999, ">100"),
        ]

        bin_data: dict[str, dict[str, Any]] = {
            label: {"count": 0, "seen": 0, "examples": []} for _, _, label in bins
        }

        for name, count in name_counts.items():
            for min_val, max_val, bin_label in bins:
                if min_val <= count <= max_val:
                    data = bin_data[bin_label]
                    data["count"] += 1
                    data["seen"] += 1
                    sample: list[tuple[str, int]] = data["examples"]

                    # Reservoir sample up to 10 names uniformly per bin.
                    if len(sample) < 10:
                        sample.append((name, count))
                    else:
                        index = random.randint(1, data["seen"])
                        if index <= 10:
                            sample[index - 1] = (name, count)
                    break

        print(f"\nTotal unique names: {len(name_counts):,}")
        total_anonid_name_pairs = sum(name_counts.values())
        print(f"Total distinct AnonID-name pairs: {total_anonid_name_pairs:,}")

        print(f"\nFrequency distribution (distinct AnonIDs per name):")
        for _, _, bin_label in bins:
            data = bin_data[bin_label]
            count = data["count"]
            if count == 0:
                print(f"  In {bin_label:>5} AnonIDs: {count:>6} names")
            else:
                examples = data["examples"]
                examples_text = ", ".join(
                    f"{example_name!r} ({example_count})"
                    for example_name, example_count in examples
                )
                print(
                    f"  In {bin_label:>5} AnonIDs: {count:>6} names  "
                    f"(random examples: {examples_text})"
                )

        print("\n" + "="*80)
        print("Rare names analysis (rows with at least one one-off full_name)")
        print("="*80)

        one_off_name_set = set(one_off_names)
        rare_combo_counts: Counter[tuple[str, ...]] = Counter()
        rare_records_total = 0
        rare_single_name_with_context = 0
        rare_single_name_with_context_and_crime = 0
        rare_single_name_with_context_and_disease = 0
        rare_single_name_with_context_seen = 0
        rare_single_name_with_context_and_crime_seen = 0
        rare_single_name_with_context_and_disease_seen = 0
        rare_single_name_with_context_examples: list[dict[str, Any]] = []
        rare_single_name_with_context_and_crime_examples: list[dict[str, Any]] = []
        rare_single_name_with_context_and_disease_examples: list[dict[str, Any]] = []
        context_labels = {"profession", "place_name", "street_city_address"}

        for query, value in zip(df["Query"], df["QueryLabels"]):
            row_labels = coerce_query_labels(value)
            labels_in_row: list[str] = []
            has_one_off_full_name = False
            query_text = "" if pd.isna(query) else str(query)

            for span in row_labels:
                if not isinstance(span, dict):
                    continue

                label = str(span.get("label", "")).strip()
                if not label:
                    continue

                labels_in_row.append(label)
                if label == "full_name":
                    text = str(span.get("text", "")).strip()
                    if text in one_off_name_set:
                        has_one_off_full_name = True

            if has_one_off_full_name and labels_in_row:
                rare_records_total += 1
                combo = tuple(sorted(labels_in_row))
                rare_combo_counts[combo] += 1

                label_counts = Counter(labels_in_row)
                has_context = any(label_counts[label] > 0 for label in context_labels)
                if label_counts["full_name"] == 1 and has_context:
                    rare_single_name_with_context += 1
                    rare_single_name_with_context_seen += 1
                    example_row = {"Query": query_text, "QueryLabels": list(row_labels)}
                    if len(rare_single_name_with_context_examples) < 50:
                        rare_single_name_with_context_examples.append(example_row)
                    else:
                        replacement_index = random.randint(1, rare_single_name_with_context_seen)
                        if replacement_index <= 50:
                            rare_single_name_with_context_examples[replacement_index - 1] = (
                                example_row
                            )
                    if label_counts["crime"] > 0:
                        rare_single_name_with_context_and_crime += 1
                        rare_single_name_with_context_and_crime_seen += 1
                        if len(rare_single_name_with_context_and_crime_examples) < 50:
                            rare_single_name_with_context_and_crime_examples.append(example_row)
                        else:
                            replacement_index = random.randint(
                                1, rare_single_name_with_context_and_crime_seen
                            )
                            if replacement_index <= 50:
                                rare_single_name_with_context_and_crime_examples[
                                    replacement_index - 1
                                ] = example_row
                    if label_counts["disease"] > 0:
                        rare_single_name_with_context_and_disease += 1
                        rare_single_name_with_context_and_disease_seen += 1
                        if len(rare_single_name_with_context_and_disease_examples) < 50:
                            rare_single_name_with_context_and_disease_examples.append(
                                example_row
                            )
                        else:
                            replacement_index = random.randint(
                                1, rare_single_name_with_context_and_disease_seen
                            )
                            if replacement_index <= 50:
                                rare_single_name_with_context_and_disease_examples[
                                    replacement_index - 1
                                ] = example_row

        if not rare_combo_counts:
            print("No rows found with one-off full_name labels.")
        else:
            print("Label combinations by frequency (including repeated labels):")
            for combo, count in rare_combo_counts.most_common():
                multiplicities = Counter(combo)
                combo_text = ", ".join(
                    f"{label} x{multiplicities[label]}"
                    for label in sorted(multiplicities.keys())
                )
                print(f"  {count:,} rows: {combo_text}")

            if rare_records_total > 0:
                frac_context = rare_single_name_with_context / rare_records_total
                frac_context_crime = rare_single_name_with_context_and_crime / rare_records_total
                frac_context_disease = (
                    rare_single_name_with_context_and_disease / rare_records_total
                )
            else:
                frac_context = 0.0
                frac_context_crime = 0.0
                frac_context_disease = 0.0

            print("\nRare-name record criteria")
            print(f"  Total rare-name records: {rare_records_total:,}")
            print(
                "  Single full_name and >=1 of "
                "[profession, place_name, street_city_address]: "
                f"{rare_single_name_with_context:,} "
                f"({frac_context:.4f} of rare-name records)"
            )
            print(
                "  Single full_name and >=1 of "
                "[profession, place_name, street_city_address] and >=1 crime: "
                f"{rare_single_name_with_context_and_crime:,} "
                f"({frac_context_crime:.4f} of rare-name records)"
            )
            print(
                "  Single full_name and >=1 of "
                "[profession, place_name, street_city_address] and >=1 disease: "
                f"{rare_single_name_with_context_and_disease:,} "
                f"({frac_context_disease:.4f} of rare-name records)"
            )

            print(
                "\n  Random examples (n=50 max) for: single full_name + "
                "[profession, place_name, street_city_address]"
            )
            unique_context_examples = dedupe_examples_by_query(rare_single_name_with_context_examples)
            for example in unique_context_examples:
                query_text = str(example.get("Query", ""))
                query_labels = example.get("QueryLabels", [])
                label_lines = format_label_type_value_pairs(query_labels)
                print(f"    Query={query_text!r}")
                if not label_lines:
                    print("      (no labels)")
                else:
                    for label_line in label_lines:
                        print(f"      {label_line}")

            print(
                "\n  Random examples (n=50 max) for: single full_name + "
                "[profession, place_name, street_city_address] + "
                "crime"
            )
            redacted_queries_crime_group: list[str] = []
            unique_crime_examples = dedupe_examples_by_query(
                rare_single_name_with_context_and_crime_examples
            )
            for example in unique_crime_examples:
                query_text = str(example.get("Query", ""))
                query_labels = example.get("QueryLabels", [])
                label_lines = format_label_type_value_pairs(query_labels)
                redacted_query = redact_full_name_in_query(query_text, query_labels)
                redacted_queries_crime_group.append(redacted_query)
                print(f"    Query={query_text!r}")
                if not label_lines:
                    print("      (no labels)")
                else:
                    for label_line in label_lines:
                        print(f"      {label_line}")

            print("\n  Redacted crime queries (one per line)")
            for redacted_query in redacted_queries_crime_group:
                print(f"    {redacted_query}")

            print(
                "\n  Random examples (n=50 max) for: single full_name + "
                "[profession, place_name, street_city_address] + "
                "disease"
            )
            redacted_queries_disease_group: list[str] = []
            unique_disease_examples = dedupe_examples_by_query(
                rare_single_name_with_context_and_disease_examples
            )
            for example in unique_disease_examples:
                query_text = str(example.get("Query", ""))
                query_labels = example.get("QueryLabels", [])
                label_lines = format_label_type_value_pairs(query_labels)
                redacted_query = redact_full_name_in_query(query_text, query_labels)
                redacted_queries_disease_group.append(redacted_query)
                print(f"    Query={query_text!r}")
                if not label_lines:
                    print("      (no labels)")
                else:
                    for label_line in label_lines:
                        print(f"      {label_line}")

            print("\n  Redacted disease queries (one per line)")
            for redacted_query in redacted_queries_disease_group:
                print(f"    {redacted_query}")

        print("\n" + "="*80)
        print("Phone number analysis")
        print("="*80)

        phone_records_total = 0
        phone_with_crime_count = 0
        phone_with_disease_count = 0
        phone_with_either_count = 0
        phone_with_crime_seen = 0
        phone_with_disease_seen = 0
        phone_with_crime_examples: list[dict[str, Any]] = []
        phone_with_disease_examples: list[dict[str, Any]] = []
        phone_example_limit = 20

        for query, value in zip(df["Query"], df["QueryLabels"]):
            row_labels = coerce_query_labels(value)
            label_counts: Counter[str] = Counter()
            for span in row_labels:
                if not isinstance(span, dict):
                    continue
                label = str(span.get("label", "")).strip()
                if not label:
                    continue
                label_counts[label] += 1

            if label_counts["phone_number"] <= 0:
                continue

            phone_records_total += 1
            has_crime = label_counts["crime"] > 0
            has_disease = label_counts["disease"] > 0
            if has_crime or has_disease:
                phone_with_either_count += 1

            query_text = "" if pd.isna(query) else str(query)
            example_row = {"Query": query_text, "QueryLabels": list(row_labels)}

            if has_crime:
                phone_with_crime_count += 1
                phone_with_crime_seen += 1
                if len(phone_with_crime_examples) < phone_example_limit:
                    phone_with_crime_examples.append(example_row)
                else:
                    replacement_index = random.randint(1, phone_with_crime_seen)
                    if replacement_index <= phone_example_limit:
                        phone_with_crime_examples[replacement_index - 1] = example_row

            if has_disease:
                phone_with_disease_count += 1
                phone_with_disease_seen += 1
                if len(phone_with_disease_examples) < phone_example_limit:
                    phone_with_disease_examples.append(example_row)
                else:
                    replacement_index = random.randint(1, phone_with_disease_seen)
                    if replacement_index <= phone_example_limit:
                        phone_with_disease_examples[replacement_index - 1] = example_row

        if phone_records_total <= 0:
            print("No rows found with phone_number labels.")
        else:
            print(f"  Total rows with phone_number: {phone_records_total:,}")
            print(
                "  Rows with phone_number and >=1 of [crime, disease]: "
                f"{phone_with_either_count:,} "
                f"({phone_with_either_count / phone_records_total:.4f} of phone_number rows)"
            )
            print(
                "  Rows with phone_number and crime: "
                f"{phone_with_crime_count:,} "
                f"({phone_with_crime_count / phone_records_total:.4f} of phone_number rows)"
            )
            print(
                "  Rows with phone_number and disease: "
                f"{phone_with_disease_count:,} "
                f"({phone_with_disease_count / phone_records_total:.4f} of phone_number rows)"
            )

            print(
                "\n  Examples (n=20 max) for: phone_number + crime"
            )
            unique_phone_crime_examples = dedupe_examples_by_query(phone_with_crime_examples)
            for example in unique_phone_crime_examples:
                query_text = str(example.get("Query", ""))
                query_labels = example.get("QueryLabels", [])
                label_lines = format_label_type_value_pairs(query_labels)
                print(f"    Query={query_text!r}")
                if not label_lines:
                    print("      (no labels)")
                else:
                    for label_line in label_lines:
                        print(f"      {label_line}")

            print(
                "\n  Examples (n=20 max) for: phone_number + disease"
            )
            unique_phone_disease_examples = dedupe_examples_by_query(phone_with_disease_examples)
            for example in unique_phone_disease_examples:
                query_text = str(example.get("Query", ""))
                query_labels = example.get("QueryLabels", [])
                label_lines = format_label_type_value_pairs(query_labels)
                print(f"    Query={query_text!r}")
                if not label_lines:
                    print("      (no labels)")
                else:
                    for label_line in label_lines:
                        print(f"      {label_line}")


def main() -> None:
    args = parse_args()
    mode = args.mode.strip().lower()

    if mode == "make_distinct":
        build_distinct_queries(args.raw_path, args.distinct_path)
        return

    if mode == "sample":
        run_sample_labeling(
            distinct_path=args.distinct_path,
            sample_output_path=args.sample_output_path,
            model_id=args.model_id,
            threshold=args.threshold,
            batch_size=args.batch_size,
            full_name_threshold=args.full_name_threshold,
            sample_size=args.sample_size,
        )
        return

    if mode == "create":
        create_labeled_parquet(
            raw_path=args.raw_path,
            label_work_dir=args.label_work_dir,
            output_path=args.output_path,
            num_chunks=args.num_chunks,
        )
        return

    if mode == "analyze":
        run_analyze_labels(output_path=args.output_path)
        return

    try:
        chunk_index = int(mode)
    except ValueError as exc:
        raise ValueError(
            "Mode must be make_distinct, sample, create, analyze, or an integer chunk index."
        ) from exc

    run_chunk_labeling(
        chunk_index=chunk_index,
        distinct_path=args.distinct_path,
        label_work_dir=args.label_work_dir,
        num_chunks=args.num_chunks,
        model_id=args.model_id,
        threshold=args.threshold,
        batch_size=args.batch_size,
        full_name_threshold=args.full_name_threshold,
    )


if __name__ == "__main__":
    main()
