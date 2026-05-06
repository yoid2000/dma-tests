"""
Build labeled query-pair training data for user-linkage modeling.

Inputs:
- labeled.parquet with columns: AnonID, Query, QueryTime, ClickURL, QueryLabels
- linkage_work/train_users.parquet from linkage_prepare.py

Outputs (under --out-dir):
- pair_queries.parquet: sampled training queries with row_id
- train_pairs.parquet: pair labels and engineered features

The script creates positive pairs (same AnonID) and hard negatives (different
AnonID, preferentially same rounded day and with token overlap).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from linkage_feature_utils import (
    build_label_pair_features,
    build_label_summary,
    coerce_query_labels,
    extract_domain,
    tokenize,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LABELED_PATH = BASE_DIR / "labeled.parquet"
DEFAULT_TRAIN_USERS_PATH = BASE_DIR / "linkage_work" / "train_users.parquet"
DEFAULT_OUT_DIR = BASE_DIR / "linkage_work"


@dataclass
class PairConfig:
    positive_pairs_per_user: int
    negative_to_positive_ratio: float
    hard_negative_try_count: int
    random_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build positive and hard-negative query pairs for linkage training."
    )
    parser.add_argument(
        "--labeled-path",
        "--raw-path",
        dest="labeled_path",
        type=Path,
        default=DEFAULT_LABELED_PATH,
        help=f"Path to labeled.parquet (default: {DEFAULT_LABELED_PATH}).",
    )
    parser.add_argument(
        "--train-users-path",
        type=Path,
        default=DEFAULT_TRAIN_USERS_PATH,
        help=f"Path to train_users.parquet (default: {DEFAULT_TRAIN_USERS_PATH}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--max-queries-per-user",
        type=int,
        default=300,
        help="Cap per-user queries to keep pair generation bounded.",
    )
    parser.add_argument(
        "--positive-pairs-per-user",
        type=int,
        default=200,
        help="Max positive pairs sampled per user.",
    )
    parser.add_argument(
        "--negative-to-positive-ratio",
        type=float,
        default=2.0,
        help="How many negatives to sample per positive pair.",
    )
    parser.add_argument(
        "--hard-negative-try-count",
        type=int,
        default=24,
        help="Number of same-day candidates to probe for hard-negative overlap.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed.",
    )
    return parser.parse_args()


def load_train_users(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"train users file not found: {path}")

    train_users = pd.read_parquet(path, columns=["AnonID"])
    ids = train_users["AnonID"].dropna().astype("int64").unique()
    if ids.size == 0:
        raise ValueError("No train users found.")
    return ids


def load_train_queries(labeled_path: Path, train_user_ids: np.ndarray) -> pd.DataFrame:
    if not labeled_path.exists():
        raise FileNotFoundError(f"labeled parquet not found: {labeled_path}")

    dataset = ds.dataset(labeled_path.as_posix(), format="parquet")
    filter_expr = pc.is_in(ds.field("AnonID"), value_set=pa_array_int64(train_user_ids))

    table = dataset.to_table(
        columns=["AnonID", "Query", "QueryTime", "ClickURL", "QueryLabels"],
        filter=filter_expr,
    )
    df = table.to_pandas(types_mapper=pd.ArrowDtype)

    df["AnonID"] = pd.to_numeric(df["AnonID"], errors="coerce").astype("Int64")
    df["Query"] = df["Query"].astype("string")
    df["QueryTime"] = pd.to_datetime(df["QueryTime"], errors="coerce")
    df["ClickURL"] = df["ClickURL"].astype("string").fillna("")
    if "QueryLabels" in df.columns:
        df["QueryLabels"] = df["QueryLabels"].apply(coerce_query_labels)
    else:
        df["QueryLabels"] = [[] for _ in range(len(df))]

    df = df.dropna(subset=["AnonID", "Query", "QueryTime"]).copy()
    df["AnonID"] = df["AnonID"].astype("int64")
    df = df[df["Query"].str.strip().str.len() > 0].copy()

    df = df.sort_values(["AnonID", "QueryTime"], kind="mergesort").reset_index(drop=True)
    return df


def pa_array_int64(values: np.ndarray):
    # Imported lazily to avoid an unconditional pyarrow array dependency at import time.
    import pyarrow as pa

    return pa.array(values, type=pa.int64())


def cap_queries_per_user(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if cap <= 0:
        raise ValueError("--max-queries-per-user must be > 0")

    df = df.copy()
    df["_rank"] = df.groupby("AnonID").cumcount()
    df = df[df["_rank"] < cap].drop(columns=["_rank"]).reset_index(drop=True)
    return df


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def sample_positive_pairs(
    user_indices: np.ndarray,
    max_pairs: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    n = user_indices.size
    if n < 2:
        return []

    pairs: set[tuple[int, int]] = set()

    # Always include adjacent pairs first: they are high-signal behaviorally.
    for i in range(n - 1):
        a = int(user_indices[i])
        b = int(user_indices[i + 1])
        if a < b:
            pairs.add((a, b))
        else:
            pairs.add((b, a))
        if len(pairs) >= max_pairs:
            return list(pairs)

    max_combinations = n * (n - 1) // 2
    target = min(max_pairs, max_combinations)

    while len(pairs) < target:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        a = int(user_indices[i])
        b = int(user_indices[j])
        if a == b:
            continue
        if a < b:
            pairs.add((a, b))
        else:
            pairs.add((b, a))

    return list(pairs)


def pick_hard_negative(
    anchor_idx: int,
    anon_ids: np.ndarray,
    day_to_indices: dict[pd.Timestamp, np.ndarray],
    query_day: pd.Timestamp,
    token_sets: list[set[str]],
    rng: np.random.Generator,
    try_count: int,
) -> int | None:
    anchor_user = anon_ids[anchor_idx]
    same_day = day_to_indices.get(query_day)

    if same_day is not None and same_day.size > 0:
        if same_day.size <= try_count:
            candidates = same_day
        else:
            candidate_positions = rng.integers(0, same_day.size, size=try_count)
            candidates = same_day[candidate_positions]

        best_idx: int | None = None
        best_score = -1.0
        anchor_tokens = token_sets[anchor_idx]

        for cand in candidates:
            cand_i = int(cand)
            if anon_ids[cand_i] == anchor_user:
                continue
            score = jaccard(anchor_tokens, token_sets[cand_i])
            if score > best_score:
                best_idx = cand_i
                best_score = score

        if best_idx is not None:
            return best_idx

    # Fallback random different user.
    n = anon_ids.size
    for _ in range(50):
        idx = int(rng.integers(0, n))
        if anon_ids[idx] != anchor_user:
            return idx
    return None


def build_pair_features(
    idx_a: int,
    idx_b: int,
    anon_ids: np.ndarray,
    queries: np.ndarray,
    query_days: np.ndarray,
    token_sets: list[set[str]],
    click_urls: np.ndarray,
    label_summaries: list,
    label: int,
) -> dict[str, int | float | str]:
    qa = str(queries[idx_a])
    qb = str(queries[idx_b])

    ta = token_sets[idx_a]
    tb = token_sets[idx_b]

    len_a = len(qa)
    len_b = len(qb)

    day_a = query_days[idx_a]
    day_b = query_days[idx_b]
    day_gap = int(abs((day_a - day_b) / np.timedelta64(1, "D")))

    inter = len(ta & tb)
    union = len(ta | tb)

    url_a = str(click_urls[idx_a])
    url_b = str(click_urls[idx_b])
    has_url_a = int(bool(url_a))
    has_url_b = int(bool(url_b))
    both_have_url = int(has_url_a and has_url_b)
    url_exact_match = int(both_have_url and url_a == url_b)
    domain_a = extract_domain(url_a)
    domain_b = extract_domain(url_b)
    url_domain_match = int(bool(domain_a) and bool(domain_b) and domain_a == domain_b)

    row = {
        "row_id_a": int(idx_a),
        "row_id_b": int(idx_b),
        "label": int(label),
        "anon_id_a": int(anon_ids[idx_a]),
        "anon_id_b": int(anon_ids[idx_b]),
        "same_day": int(day_gap == 0),
        "day_gap": int(day_gap),
        "len_a": int(len_a),
        "len_b": int(len_b),
        "len_abs_diff": int(abs(len_a - len_b)),
        "token_overlap": int(inter),
        "token_union": int(union),
        "token_jaccard": float(inter / union) if union else 0.0,
        "prefix3_equal": int(qa[:3].lower() == qb[:3].lower()),
        "has_digit_a": int(any(ch.isdigit() for ch in qa)),
        "has_digit_b": int(any(ch.isdigit() for ch in qb)),
        "exact_query_match": int(qa.strip().lower() == qb.strip().lower()),
        "has_url_a": has_url_a,
        "has_url_b": has_url_b,
        "both_have_url": both_have_url,
        "url_exact_match": url_exact_match,
        "url_domain_match": url_domain_match,
    }
    row.update(build_label_pair_features(label_summaries[idx_a], label_summaries[idx_b]))
    return row


def main() -> None:
    args = parse_args()

    if args.positive_pairs_per_user <= 0:
        raise ValueError("--positive-pairs-per-user must be > 0")
    if args.negative_to_positive_ratio <= 0:
        raise ValueError("--negative-to-positive-ratio must be > 0")
    if args.hard_negative_try_count <= 0:
        raise ValueError("--hard-negative-try-count must be > 0")

    config = PairConfig(
        positive_pairs_per_user=args.positive_pairs_per_user,
        negative_to_positive_ratio=args.negative_to_positive_ratio,
        hard_negative_try_count=args.hard_negative_try_count,
        random_seed=args.seed,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading train users from: {args.train_users_path}")
    train_user_ids = load_train_users(args.train_users_path)
    print(f"Loaded train users: {train_user_ids.size:,}")

    print(f"Loading train queries from: {args.labeled_path}")
    df = load_train_queries(args.labeled_path, train_user_ids)
    print(f"Queried rows for train users: {len(df):,}")

    print(f"Applying per-user cap: {args.max_queries_per_user}")
    df = cap_queries_per_user(df, args.max_queries_per_user)
    print(f"Rows after cap: {len(df):,}")

    df = df.sort_values(["AnonID", "QueryTime"], kind="mergesort").reset_index(drop=True)
    df["QueryDay"] = df["QueryTime"].dt.floor("D")
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    anon_ids = df["AnonID"].to_numpy(dtype=np.int64)
    queries = df["Query"].astype("string").to_numpy()
    query_days = df["QueryDay"].to_numpy(dtype="datetime64[D]")
    click_urls = df["ClickURL"].fillna("").astype(str).to_numpy()
    query_labels = df["QueryLabels"].tolist()

    token_sets = [tokenize(str(text)) for text in queries]
    label_summaries = [build_label_summary(coerce_query_labels(value)) for value in query_labels]

    day_values = pd.Series(df["QueryDay"].to_numpy()).astype("datetime64[ns]")
    day_to_indices: dict[pd.Timestamp, np.ndarray] = {}
    for day, idxs in day_values.groupby(day_values).groups.items():
        day_to_indices[pd.Timestamp(day)] = np.asarray(list(idxs), dtype=np.int64)

    rng = np.random.default_rng(config.random_seed)

    print("Sampling positive and hard-negative pairs")
    rows: list[dict[str, int | float | str]] = []
    positives = 0
    negatives = 0

    user_groups = df.groupby("AnonID", sort=False).groups
    for user_id, indices in user_groups.items():
        user_indices = np.asarray(list(indices), dtype=np.int64)
        pos_pairs = sample_positive_pairs(
            user_indices=user_indices,
            max_pairs=config.positive_pairs_per_user,
            rng=rng,
        )

        for idx_a, idx_b in pos_pairs:
            rows.append(
                build_pair_features(
                    idx_a=idx_a,
                    idx_b=idx_b,
                    anon_ids=anon_ids,
                    queries=queries,
                    query_days=query_days,
                    token_sets=token_sets,
                    click_urls=click_urls,
                    label_summaries=label_summaries,
                    label=1,
                )
            )
            positives += 1

            n_neg = int(round(config.negative_to_positive_ratio))
            for _ in range(n_neg):
                anchor = idx_a if rng.random() < 0.5 else idx_b
                qday = pd.Timestamp(query_days[anchor])
                neg_idx = pick_hard_negative(
                    anchor_idx=anchor,
                    anon_ids=anon_ids,
                    day_to_indices=day_to_indices,
                    query_day=qday,
                    token_sets=token_sets,
                    rng=rng,
                    try_count=config.hard_negative_try_count,
                )
                if neg_idx is None:
                    continue

                a, b = (anchor, neg_idx) if anchor < neg_idx else (neg_idx, anchor)
                rows.append(
                    build_pair_features(
                        idx_a=a,
                        idx_b=b,
                        anon_ids=anon_ids,
                        queries=queries,
                        query_days=query_days,
                        token_sets=token_sets,
                        click_urls=click_urls,
                        label_summaries=label_summaries,
                        label=0,
                    )
                )
                negatives += 1

    if not rows:
        raise ValueError("No training pairs were generated. Check split sizes and caps.")

    pairs_df = pd.DataFrame(rows)

    # Drop accidental duplicate pair-label rows.
    pairs_df = pairs_df.drop_duplicates(
        subset=["row_id_a", "row_id_b", "label"], keep="first"
    ).reset_index(drop=True)

    query_out = df[
        ["row_id", "AnonID", "Query", "QueryTime", "QueryDay", "ClickURL", "QueryLabels"]
    ].copy()

    query_path = out_dir / "pair_queries.parquet"
    pairs_path = out_dir / "train_pairs.parquet"

    query_out.to_parquet(query_path, index=False)
    pairs_df.to_parquet(pairs_path, index=False)

    print(f"Wrote: {query_path}")
    print(f"Wrote: {pairs_path}")
    print(f"Positive pairs: {positives:,}")
    print(f"Negative pairs: {negatives:,}")
    print(f"Final rows in train_pairs.parquet: {len(pairs_df):,}")


if __name__ == "__main__":
    main()
