"""
Distributed labeling utility for AOL query data.

Modes:
- make_distinct: build distinct_queries.parquet from raw.parquet Query values
- <i>: label chunk i (0-based) of distinct queries and write label_work/i.parquet
- create: combine label_work/*.parquet with raw.parquet into labeled.parquet
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
from typing import Any

from email_validator import EmailNotValidError, validate_email
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distributed labeler for raw.parquet. Modes: make_distinct, sample, "
            "<chunk_index>, create."
        )
    )
    parser.add_argument(
        "mode",
        help="One of: make_distinct, sample, create, or chunk index i (0-based integer).",
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
    missing_files = [str(path) for path in expected_files if not path.exists()]
    if missing_files:
        preview = ", ".join(missing_files[:10])
        suffix = "..." if len(missing_files) > 10 else ""
        raise FileNotFoundError(
            f"Missing {len(missing_files)} chunk files in {label_work_dir}: {preview}{suffix}"
        )

    label_frames = [pd.read_parquet(path, columns=["Query", "QueryLabels"]) for path in expected_files]
    labels_df = pd.concat(label_frames, ignore_index=True)
    labels_df = labels_df.drop_duplicates(subset=["Query"], keep="last")

    print(f"Loading raw data from: {raw_path}")
    raw_df = pd.read_parquet(raw_path)
    labeled_df = raw_df.merge(labels_df, on="Query", how="left")
    labeled_df["QueryLabels"] = labeled_df["QueryLabels"].apply(
        lambda value: value if isinstance(value, list) else []
    )

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

    try:
        chunk_index = int(mode)
    except ValueError as exc:
        raise ValueError(
            "Mode must be make_distinct, sample, create, or an integer chunk index."
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
