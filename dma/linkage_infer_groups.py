"""
Run constrained group inference for the 50-user linkage setting.

This script:
1. Loads sampled groups from linkage_work/group_manifest.parquet.
2. Pulls each group's queries from raw.parquet.
3. Builds pair features matching linkage_build_pairs.py.
4. Scores same-user probabilities with linkage_pair_model.pkl.
5. Clusters each group to a target number of clusters (default 50).
6. Writes per-query assignments with confidence signals.

Outputs:
- group_query_assignments.parquet
- group_cluster_summary.parquet
- group_run_summary.parquet
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import re
import urllib.parse

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_PATH = BASE_DIR / "raw.parquet"
DEFAULT_GROUP_MANIFEST_PATH = BASE_DIR / "linkage_work" / "group_manifest.parquet"
DEFAULT_MODEL_PATH = BASE_DIR / "linkage_work" / "model" / "linkage_pair_model.pkl"
DEFAULT_OUT_DIR = BASE_DIR / "linkage_work" / "inference"

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def apply_calibrator(calibrator, base_proba: np.ndarray) -> np.ndarray:
    # Support new sigmoid calibrator (predict_proba) and legacy isotonic (predict).
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(base_proba.reshape(-1, 1))[:, 1]
    if hasattr(calibrator, "predict"):
        return calibrator.predict(base_proba)
    raise TypeError("Unsupported calibrator object: missing predict/predict_proba")


def _checkpoint_stem(split_name: str, group_id: int) -> str:
    return f"{split_name}_{group_id:06d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run query-to-cluster inference over sampled 50-user groups using a "
            "trained pairwise model."
        )
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help=f"Path to raw.parquet (default: {DEFAULT_RAW_PATH}).",
    )
    parser.add_argument(
        "--group-manifest-path",
        type=Path,
        default=DEFAULT_GROUP_MANIFEST_PATH,
        help=f"Path to group_manifest.parquet (default: {DEFAULT_GROUP_MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to linkage_pair_model.pkl (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test", "all"],
        help="Which manifest split to run.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="Optional cap on number of groups to process (0 = all).",
    )
    parser.add_argument(
        "--max-queries-per-user",
        type=int,
        default=100,
        help="Cap per-user queries per group for compute control.",
    )
    parser.add_argument(
        "--target-clusters",
        type=int,
        default=50,
        help="Target number of output clusters per group.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help=(
            "Skip inference; merge all existing per-group checkpoint files under "
            "out-dir/groups/ into the final output parquets and exit."
        ),
    )
    return parser.parse_args()


def _write_group_checkpoint(
    groups_dir: Path,
    split_name: str,
    group_id: int,
    assign_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    metrics: dict,
) -> None:
    stem = _checkpoint_stem(split_name, group_id)
    assign_df.to_parquet(groups_dir / f"{stem}_assignments.parquet", index=False)
    cluster_df.to_parquet(groups_dir / f"{stem}_clusters.parquet", index=False)
    (groups_dir / f"{stem}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


def _checkpoint_exists(groups_dir: Path, split_name: str, group_id: int) -> bool:
    stem = _checkpoint_stem(split_name, group_id)
    return (groups_dir / f"{stem}_assignments.parquet").exists()


def _merge_checkpoints(
    groups_dir: Path,
    out_dir: Path,
    split: str,
    max_groups: int,
    target_clusters: int,
    max_queries_per_user: int,
) -> None:
    assign_files = sorted(groups_dir.glob("*_assignments.parquet"))
    cluster_files = sorted(groups_dir.glob("*_clusters.parquet"))
    metric_files = sorted(groups_dir.glob("*_metrics.json"))

    if not assign_files:
        print("No checkpoint files found to merge.")
        return

    assignments = pd.concat(
        [pd.read_parquet(f) for f in assign_files], ignore_index=True
    )
    clusters = pd.concat(
        [pd.read_parquet(f) for f in cluster_files], ignore_index=True
    )
    run_rows = [json.loads(f.read_text(encoding="utf-8")) for f in metric_files]
    summary = pd.DataFrame(run_rows)

    assignments_path = out_dir / "group_query_assignments.parquet"
    clusters_path = out_dir / "group_cluster_summary.parquet"
    summary_path = out_dir / "group_run_summary.parquet"
    json_path = out_dir / "group_run_summary.json"

    assignments.to_parquet(assignments_path, index=False)
    clusters.to_parquet(clusters_path, index=False)
    summary.to_parquet(summary_path, index=False)

    payload = {
        "groups": int(len(summary)),
        "mean_queries_per_group": float(summary["queries"].mean()) if not summary.empty else 0.0,
        "mean_ari": float(summary["ari"].dropna().mean()) if (not summary.empty and summary["ari"].notna().any()) else None,
        "target_clusters": target_clusters,
        "max_queries_per_user": max_queries_per_user,
        "split": split,
        "max_groups": max_groups,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Merged {len(assign_files)} checkpoint(s).")
    print(f"Wrote: {assignments_path}")
    print(f"Wrote: {clusters_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {json_path}")
    print(json.dumps(payload, indent=2))


def pa_array_int64(values: np.ndarray):
    import pyarrow as pa

    return pa.array(values, type=pa.int64())


def load_model(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    with path.open("rb") as f:
        artifact = pickle.load(f)

    required = {"model", "calibrator", "feature_columns"}
    missing = required - set(artifact.keys())
    if missing:
        raise ValueError(f"Model artifact missing required keys: {sorted(missing)}")
    return artifact


def load_manifest(path: Path, split: str, max_groups: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Group manifest not found: {path}")

    manifest = pd.read_parquet(path)
    required = {"split", "group_id", "AnonID"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    if split != "all":
        manifest = manifest[manifest["split"] == split].copy()

    manifest = manifest.sort_values(["split", "group_id", "AnonID"]).reset_index(drop=True)

    if max_groups > 0:
        keep_groups = (
            manifest[["split", "group_id"]]
            .drop_duplicates()
            .head(max_groups)
            .copy()
        )
        manifest = manifest.merge(keep_groups, on=["split", "group_id"], how="inner")

    if manifest.empty:
        raise ValueError("Manifest selection is empty. Check --split and --max-groups.")

    return manifest


def load_group_queries(raw_path: Path, group_users: np.ndarray) -> pd.DataFrame:
    dataset = ds.dataset(raw_path.as_posix(), format="parquet")
    filter_expr = pc.is_in(ds.field("AnonID"), value_set=pa_array_int64(group_users))
    table = dataset.to_table(columns=["AnonID", "Query", "QueryTime", "ClickURL"], filter=filter_expr)

    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    df["AnonID"] = pd.to_numeric(df["AnonID"], errors="coerce").astype("Int64")
    df["Query"] = df["Query"].astype("string")
    df["QueryTime"] = pd.to_datetime(df["QueryTime"], errors="coerce")
    df["ClickURL"] = df["ClickURL"].astype("string").fillna("")

    df = df.dropna(subset=["AnonID", "Query", "QueryTime"]).copy()
    df["AnonID"] = df["AnonID"].astype("int64")
    df = df[df["Query"].str.strip().str.len() > 0].copy()

    df = df.sort_values(["AnonID", "QueryTime"], kind="mergesort").reset_index(drop=True)
    return df


def cap_queries_per_user(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if cap <= 0:
        raise ValueError("--max-queries-per-user must be > 0")
    out = df.copy()
    out["_rank"] = out.groupby("AnonID").cumcount()
    out = out[out["_rank"] < cap].drop(columns=["_rank"]).reset_index(drop=True)
    return out


def tokenize(text: str) -> set[str]:
    return {tok for tok in TOKEN_RE.findall(text.lower()) if tok}


def extract_domain(url: str) -> str:
    """Return the netloc of a URL, lowercased. Empty string if not parseable."""
    if not url:
        return ""
    try:
        netloc = urllib.parse.urlparse(url).netloc
        return netloc.lower()
    except Exception:
        return ""


def build_pair_feature_dict(
    idx_a: int,
    idx_b: int,
    anon_ids: np.ndarray,
    queries: np.ndarray,
    query_days: np.ndarray,
    token_sets: list[set[str]],
    click_urls: np.ndarray,
) -> dict[str, int | float]:
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

    return {
        "row_id_a": int(idx_a),
        "row_id_b": int(idx_b),
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


_PAIR_BATCH = 500_000  # pairs per scoring batch; keeps peak memory bounded


def score_probability_matrix(
    feature_columns: list[str],
    model,
    calibrator,
    anon_ids: np.ndarray,
    queries: np.ndarray,
    query_days: np.ndarray,
    click_urls: np.ndarray,
) -> np.ndarray:
    n = len(queries)
    probs = np.eye(n, dtype=np.float64)

    if n <= 1:
        return probs

    # Upper-triangle pair indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    n_pairs = len(i_idx)

    if n_pairs == 0:
        return probs

    # Pre-compute per-query derived values once
    str_queries = [str(q) for q in queries]
    token_sets = [tokenize(q) for q in str_queries]
    str_urls = [str(u) for u in click_urls]
    domains = [extract_domain(u) for u in str_urls]
    prefix3 = [q[:3].lower() for q in str_queries]
    has_digit = np.array([int(any(ch.isdigit() for ch in q)) for q in str_queries])
    query_lower = [q.strip().lower() for q in str_queries]
    has_url = np.array([int(bool(u)) for u in str_urls])

    calibrated_all = np.empty(n_pairs, dtype=np.float64)

    for start in range(0, n_pairs, _PAIR_BATCH):
        end = min(start + _PAIR_BATCH, n_pairs)
        ia = i_idx[start:end]
        ja = j_idx[start:end]
        m = end - start

        day_gap = np.abs(
            (query_days[ia] - query_days[ja]) / np.timedelta64(1, "D")
        ).astype(np.int64)

        len_a = np.array([len(str_queries[k]) for k in ia])
        len_b = np.array([len(str_queries[k]) for k in ja])

        inter = np.array([len(token_sets[ia[k]] & token_sets[ja[k]]) for k in range(m)])
        union = np.array([len(token_sets[ia[k]] | token_sets[ja[k]]) for k in range(m)])

        has_url_a = has_url[ia]
        has_url_b = has_url[ja]
        both_have = has_url_a & has_url_b

        url_exact = np.array(
            [int(both_have[k] and str_urls[ia[k]] == str_urls[ja[k]]) for k in range(m)]
        )
        url_domain = np.array(
            [
                int(
                    bool(domains[ia[k]])
                    and bool(domains[ja[k]])
                    and domains[ia[k]] == domains[ja[k]]
                )
                for k in range(m)
            ]
        )

        feat: dict[str, np.ndarray] = {
            "row_id_a": ia.astype(np.int64),
            "row_id_b": ja.astype(np.int64),
            "anon_id_a": anon_ids[ia].astype(np.int64),
            "anon_id_b": anon_ids[ja].astype(np.int64),
            "same_day": (day_gap == 0).astype(np.int64),
            "day_gap": day_gap,
            "len_a": len_a,
            "len_b": len_b,
            "len_abs_diff": np.abs(len_a - len_b),
            "token_overlap": inter,
            "token_union": union,
            "token_jaccard": np.divide(inter, union, out=np.zeros(m, dtype=np.float64), where=union > 0),
            "prefix3_equal": np.array(
                [int(prefix3[ia[k]] == prefix3[ja[k]]) for k in range(m)]
            ),
            "has_digit_a": has_digit[ia],
            "has_digit_b": has_digit[ja],
            "exact_query_match": np.array(
                [int(query_lower[ia[k]] == query_lower[ja[k]]) for k in range(m)]
            ),
            "has_url_a": has_url_a,
            "has_url_b": has_url_b,
            "both_have_url": both_have,
            "url_exact_match": url_exact,
            "url_domain_match": url_domain,
        }

        missing = [c for c in feature_columns if c not in feat]
        if missing:
            raise ValueError(f"Missing feature columns at inference time: {missing}")

        x = np.column_stack([feat[c] for c in feature_columns]).astype(np.float64)
        base = model.predict_proba(x)[:, 1]
        calibrated_all[start:end] = apply_calibrator(calibrator, base)

    probs[i_idx, j_idx] = calibrated_all
    probs[j_idx, i_idx] = calibrated_all

    return probs


def cluster_from_probabilities(probs: np.ndarray, target_clusters: int) -> np.ndarray:
    n = probs.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)

    k = min(max(1, target_clusters), n)
    distance = 1.0 - probs
    np.fill_diagonal(distance, 0.0)

    clustering = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(distance)
    return labels.astype(np.int64)


def assignment_confidence(
    probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(labels)
    unique_clusters = np.unique(labels)

    best_score = np.zeros(n, dtype=np.float64)
    second_score = np.zeros(n, dtype=np.float64)

    for i in range(n):
        scores: list[float] = []
        for c in unique_clusters:
            members = np.where(labels == c)[0]
            if members.size == 0:
                continue

            if members.size == 1 and members[0] == i:
                mean_p = 1.0
            else:
                others = members[members != i]
                if others.size == 0:
                    mean_p = 1.0
                else:
                    mean_p = float(probs[i, others].mean())
            scores.append(mean_p)

        if not scores:
            best_score[i] = 0.0
            second_score[i] = 0.0
            continue

        ordered = sorted(scores, reverse=True)
        best_score[i] = ordered[0]
        second_score[i] = ordered[1] if len(ordered) > 1 else 0.0

    margin = best_score - second_score
    return best_score, second_score, margin


def cluster_confidence(probs: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = {}
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        if members.size <= 1:
            out[int(c)] = 1.0
            continue

        tri = probs[np.ix_(members, members)]
        iu = np.triu_indices(tri.shape[0], k=1)
        vals = tri[iu]
        if vals.size == 0:
            out[int(c)] = 1.0
        else:
            out[int(c)] = float(np.quantile(vals, 0.10))
    return out


def process_group(
    split_name: str,
    group_id: int,
    group_users: np.ndarray,
    raw_path: Path,
    feature_columns: list[str],
    model,
    calibrator,
    max_queries_per_user: int,
    target_clusters: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    group_df = load_group_queries(raw_path, group_users)
    group_df = cap_queries_per_user(group_df, max_queries_per_user)

    group_df = group_df.sort_values("QueryTime", kind="mergesort").reset_index(drop=True)
    group_df["QueryDay"] = group_df["QueryTime"].dt.floor("D")

    n = len(group_df)
    if n == 0:
        empty_assign = pd.DataFrame(
            columns=[
                "split",
                "group_id",
                "query_idx",
                "Query",
                "QueryTimeRounded24h",
                "predicted_cluster",
                "assign_best_score",
                "assign_second_score",
                "assign_margin",
                "cluster_confidence_p10",
                "AnonID_true",
            ]
        )
        empty_cluster = pd.DataFrame(
            columns=["split", "group_id", "predicted_cluster", "cluster_size", "cluster_confidence_p10"]
        )
        metrics = {
            "split": split_name,
            "group_id": int(group_id),
            "queries": 0,
            "true_users": int(len(np.unique(group_users))),
            "pred_clusters": 0,
            "ari": float("nan"),
        }
        return empty_assign, empty_cluster, metrics

    anon_ids = group_df["AnonID"].to_numpy(dtype=np.int64)
    queries = group_df["Query"].astype("string").to_numpy()
    query_days = group_df["QueryDay"].to_numpy(dtype="datetime64[D]")
    click_urls = group_df["ClickURL"].fillna("").astype(str).to_numpy()

    probs = score_probability_matrix(
        feature_columns=feature_columns,
        model=model,
        calibrator=calibrator,
        anon_ids=anon_ids,
        queries=queries,
        query_days=query_days,
        click_urls=click_urls,
    )

    labels = cluster_from_probabilities(probs, target_clusters=target_clusters)
    best_score, second_score, margin = assignment_confidence(probs, labels)
    cluster_conf = cluster_confidence(probs, labels)

    assign = pd.DataFrame(
        {
            "split": split_name,
            "group_id": int(group_id),
            "query_idx": np.arange(n, dtype=np.int64),
            "Query": group_df["Query"].astype("string"),
            "QueryTimeRounded24h": group_df["QueryDay"],
            "predicted_cluster": labels,
            "assign_best_score": best_score,
            "assign_second_score": second_score,
            "assign_margin": margin,
            "cluster_confidence_p10": [cluster_conf[int(c)] for c in labels],
            "AnonID_true": anon_ids,
        }
    )

    cluster_sizes = (
        assign.groupby("predicted_cluster", as_index=False)
        .size()
        .rename(columns={"size": "cluster_size"})
    )
    cluster_sizes["split"] = split_name
    cluster_sizes["group_id"] = int(group_id)
    cluster_sizes["cluster_confidence_p10"] = cluster_sizes["predicted_cluster"].map(cluster_conf)

    true_users = np.unique(anon_ids)
    ari = float(adjusted_rand_score(anon_ids, labels)) if true_users.size > 1 else float("nan")

    metrics = {
        "split": split_name,
        "group_id": int(group_id),
        "queries": int(n),
        "true_users": int(true_users.size),
        "pred_clusters": int(np.unique(labels).size),
        "ari": ari,
    }

    return assign, cluster_sizes, metrics


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = out_dir / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        print("--merge-only: merging existing checkpoints.")
        _merge_checkpoints(
            groups_dir=groups_dir,
            out_dir=out_dir,
            split=args.split,
            max_groups=args.max_groups,
            target_clusters=args.target_clusters,
            max_queries_per_user=args.max_queries_per_user,
        )
        return

    if not args.raw_path.exists():
        raise FileNotFoundError(f"raw.parquet not found: {args.raw_path}")

    artifact = load_model(args.model_path)
    model = artifact["model"]
    calibrator = artifact["calibrator"]
    feature_columns = artifact["feature_columns"]

    manifest = load_manifest(
        path=args.group_manifest_path,
        split=args.split,
        max_groups=args.max_groups,
    )

    groups = manifest[["split", "group_id"]].drop_duplicates().reset_index(drop=True)
    n_total = len(groups)
    n_skipped = 0

    print(f"Processing groups: {n_total:,} (checkpoints in: {groups_dir})")
    for idx, row in groups.iterrows():
        split_name = str(row["split"])
        group_id = int(row["group_id"])

        if _checkpoint_exists(groups_dir, split_name, group_id):
            n_skipped += 1
            continue

        user_ids = (
            manifest[(manifest["split"] == split_name) & (manifest["group_id"] == group_id)]["AnonID"]
            .dropna()
            .astype("int64")
            .unique()
        )

        assign_df, cluster_df, metrics = process_group(
            split_name=split_name,
            group_id=group_id,
            group_users=user_ids,
            raw_path=args.raw_path,
            feature_columns=feature_columns,
            model=model,
            calibrator=calibrator,
            max_queries_per_user=args.max_queries_per_user,
            target_clusters=args.target_clusters,
        )

        _write_group_checkpoint(
            groups_dir=groups_dir,
            split_name=split_name,
            group_id=group_id,
            assign_df=assign_df,
            cluster_df=cluster_df,
            metrics=metrics,
        )

        done = idx + 1
        if done % 5 == 0 or done == n_total:
            print(f"Processed {done - n_skipped:,} new / {done:,} total of {n_total:,} groups")

    if n_skipped:
        print(f"Skipped {n_skipped:,} already-checkpointed groups.")

    print("\nMerging checkpoints into final outputs...")
    _merge_checkpoints(
        groups_dir=groups_dir,
        out_dir=out_dir,
        split=args.split,
        max_groups=args.max_groups,
        target_clusters=args.target_clusters,
        max_queries_per_user=args.max_queries_per_user,
    )


if __name__ == "__main__":
    main()
