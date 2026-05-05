"""
Filter group inference results to high-confidence query assignments.

Input:
- group_query_assignments.parquet from linkage_infer_groups.py

Outputs (under --out-dir):
- confident_assignments.parquet: filtered high-confidence query assignments
- confident_threshold_sweep.parquet: precision/recall at a range of thresholds
- confident_threshold_sweep.json: same summary as JSON

Filters on:
- assign_margin: best_cluster_score - second_cluster_score
- cluster_confidence_p10: 10th-percentile internal pair probability of assigned cluster

If AnonID_true is present in the input, precision/recall are computed against it.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ASSIGNMENTS_PATH = BASE_DIR / "linkage_work" / "inference" / "group_query_assignments.parquet"
DEFAULT_OUT_DIR = BASE_DIR / "linkage_work" / "inference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter linkage assignments to high-confidence outputs and sweep thresholds."
        )
    )
    parser.add_argument(
        "--assignments-path",
        type=Path,
        default=DEFAULT_ASSIGNMENTS_PATH,
        help=f"Path to group_query_assignments.parquet (default: {DEFAULT_ASSIGNMENTS_PATH}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.3,
        help="Minimum assign_margin for a query to be included (default: 0.3).",
    )
    parser.add_argument(
        "--cluster-confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum cluster_confidence_p10 for a query to be included (default: 0.5).",
    )
    parser.add_argument(
        "--sweep-steps",
        type=int,
        default=20,
        help="Number of threshold steps for precision/recall sweep.",
    )
    parser.add_argument(
        "--margin-sweep-max",
        type=float,
        default=None,
        help=(
            "Optional explicit max margin threshold for the sweep. "
            "Use this to force exploration of high-margin, high-precision regions "
            "(e.g. 0.03). If omitted, sweep max is chosen adaptively."
        ),
    )
    parser.add_argument(
        "--print-pair-examples",
        type=int,
        default=0,
        help=(
            "If > 0, print up to this many example confident pairs from the filtered "
            "output and write them to confident_pair_examples.parquet."
        ),
    )
    return parser.parse_args()


def load_assignments(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Assignments file not found: {path}")

    df = pd.read_parquet(path)
    required = {"split", "group_id", "query_idx", "predicted_cluster", "assign_margin", "cluster_confidence_p10"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Assignments file missing required columns: {sorted(missing)}")

    return df


def has_ground_truth(df: pd.DataFrame) -> bool:
    return "AnonID_true" in df.columns


def compute_pairwise_precision_recall(df: pd.DataFrame) -> tuple[float, float]:
    """
    Compute pairwise precision and recall for same-user pair predictions.
    Within each (split, group_id):
    - predicted positive pair: two queries assigned to same predicted_cluster
    - true positive pair: two queries from same AnonID_true
    """
    total_tp = 0
    total_pred_pos = 0
    total_true_pos = 0

    for (split, gid), grp in df.groupby(["split", "group_id"], sort=False):
        n = len(grp)
        if n < 2:
            continue

        pred_clusters = grp["predicted_cluster"].to_numpy()
        true_ids = grp["AnonID_true"].to_numpy()

        for i in range(n):
            for j in range(i + 1, n):
                pred_same = pred_clusters[i] == pred_clusters[j]
                true_same = true_ids[i] == true_ids[j]

                if pred_same:
                    total_pred_pos += 1
                    if true_same:
                        total_tp += 1
                if true_same:
                    total_true_pos += 1

    precision = total_tp / total_pred_pos if total_pred_pos > 0 else float("nan")
    recall = total_tp / total_true_pos if total_true_pos > 0 else float("nan")
    return precision, recall


def apply_threshold(df: pd.DataFrame, margin_t: float, cluster_t: float) -> pd.DataFrame:
    mask = (df["assign_margin"] >= margin_t) & (df["cluster_confidence_p10"] >= cluster_t)
    return df.loc[mask].copy()


def compute_pairwise_pr_fast(df: pd.DataFrame) -> tuple[float, float, int, int]:
    """
    Faster pairwise computation using per-group cluster/AnonID membership sets.
    Returns precision, recall, pred_positives, true_positives.
    """
    total_tp = 0
    total_pred_pos = 0
    total_true_pos = 0

    for (split, gid), grp in df.groupby(["split", "group_id"], sort=False):
        # Count pairs within same predicted cluster (predicted positives).
        pred_cluster_sizes = grp.groupby("predicted_cluster").size()
        pred_pos = int(((pred_cluster_sizes * (pred_cluster_sizes - 1)) / 2).sum())

        # Count pairs within same AnonID_true (true positives in the universe).
        true_id_sizes = grp.groupby("AnonID_true").size()
        true_pos_universe = int(((true_id_sizes * (true_id_sizes - 1)) / 2).sum())

        # True positives: pairs that are both same predicted cluster AND same AnonID.
        grp2 = grp[["predicted_cluster", "AnonID_true"]].copy()
        co = grp2.groupby(["predicted_cluster", "AnonID_true"]).size()
        tp = int(((co * (co - 1)) / 2).sum())

        total_pred_pos += pred_pos
        total_true_pos += true_pos_universe
        total_tp += tp

    precision = total_tp / total_pred_pos if total_pred_pos > 0 else float("nan")
    recall = total_tp / total_true_pos if total_true_pos > 0 else float("nan")
    return precision, recall, total_pred_pos, total_tp


def compute_pairwise_counts(df: pd.DataFrame) -> tuple[int, int, int]:
    """
    Return aggregate pairwise counts over all groups:
    - true positives (same predicted cluster and same AnonID_true)
    - predicted positives (same predicted cluster)
    - true positives in universe (same AnonID_true)
    """
    total_tp = 0
    total_pred_pos = 0
    total_true_pos = 0

    for _, grp in df.groupby(["split", "group_id"], sort=False):
        pred_cluster_sizes = grp.groupby("predicted_cluster").size()
        pred_pos = int(((pred_cluster_sizes * (pred_cluster_sizes - 1)) / 2).sum())

        true_id_sizes = grp.groupby("AnonID_true").size()
        true_pos_universe = int(((true_id_sizes * (true_id_sizes - 1)) / 2).sum())

        co = grp.groupby(["predicted_cluster", "AnonID_true"]).size()
        tp = int(((co * (co - 1)) / 2).sum())

        total_pred_pos += pred_pos
        total_true_pos += true_pos_universe
        total_tp += tp

    return total_tp, total_pred_pos, total_true_pos


def compute_groupwise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for (split, gid), grp in df.groupby(["split", "group_id"], sort=False):
        pred_cluster_sizes = grp.groupby("predicted_cluster").size()
        pred_pos = int(((pred_cluster_sizes * (pred_cluster_sizes - 1)) / 2).sum())

        true_id_sizes = grp.groupby("AnonID_true").size()
        true_pos_universe = int(((true_id_sizes * (true_id_sizes - 1)) / 2).sum())

        co = grp.groupby(["predicted_cluster", "AnonID_true"]).size()
        tp = int(((co * (co - 1)) / 2).sum())

        precision = tp / pred_pos if pred_pos > 0 else float("nan")
        recall = tp / true_pos_universe if true_pos_universe > 0 else float("nan")

        rows.append(
            {
                "split": str(split),
                "group_id": int(gid),
                "pairwise_precision": float(precision),
                "pairwise_recall": float(recall),
            }
        )

    return pd.DataFrame(rows)


def percentile_triplet(values: pd.Series) -> tuple[float, float, float]:
    valid = values.dropna()
    if valid.empty:
        nan = float("nan")
        return nan, nan, nan
    return (
        float(np.quantile(valid, 0.10)),
        float(np.quantile(valid, 0.50)),
        float(np.quantile(valid, 0.90)),
    )


def adaptive_sweep_values(values: pd.Series, sweep_steps: int, q_hi: float = 0.995) -> np.ndarray:
    """Build threshold values using observed score distribution with a baseline 0.0."""
    if sweep_steps <= 0:
        raise ValueError("--sweep-steps must be > 0")

    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.zeros(sweep_steps, dtype=np.float64)

    hi = float(np.quantile(clean, q_hi))
    max_observed = float(clean.max())
    hi = min(max(hi, 0.0), max_observed)

    if sweep_steps == 1:
        return np.array([0.0], dtype=np.float64)

    out = np.linspace(0.0, hi, sweep_steps, dtype=np.float64)
    out[0] = 0.0
    return out


def adaptive_high_tail_sweep_values(
    values: pd.Series,
    sweep_steps: int,
    q_lo: float = 0.90,
    q_hi: float = 0.999,
) -> np.ndarray:
    """
    Build threshold values focused on the high end of the observed distribution.
    Includes 0.0 as the first baseline threshold.
    """
    if sweep_steps <= 0:
        raise ValueError("--sweep-steps must be > 0")

    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.zeros(sweep_steps, dtype=np.float64)

    if sweep_steps == 1:
        return np.array([0.0], dtype=np.float64)

    lo = float(np.quantile(clean, q_lo))
    hi = float(np.quantile(clean, q_hi))
    max_observed = float(clean.max())
    lo = min(max(lo, 0.0), max_observed)
    hi = min(max(hi, lo), max_observed)

    if hi == lo:
        min_observed = float(clean.min())
        if max_observed > min_observed:
            tail = np.linspace(min_observed, max_observed, sweep_steps - 1, dtype=np.float64)
            return np.concatenate(([0.0], tail))

        # Fully degenerate distribution: repeat same threshold values after baseline.
        out = np.full(sweep_steps, lo, dtype=np.float64)
        out[0] = 0.0
        return out

    tail = np.linspace(lo, hi, sweep_steps - 1, dtype=np.float64)
    out = np.concatenate(([0.0], tail))
    return out


def build_threshold_sweep(
    df: pd.DataFrame,
    sweep_steps: int,
    gt_available: bool,
    full_true_pos_universe: int,
    margin_sweep_max: float | None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    margin_clean = pd.to_numeric(df["assign_margin"], errors="coerce").dropna()
    observed_margin_max = float(margin_clean.max()) if not margin_clean.empty else 0.0
    if margin_sweep_max is None:
        margin_vals = adaptive_sweep_values(df["assign_margin"], sweep_steps=sweep_steps, q_hi=0.995)
    else:
        forced_max = max(0.0, float(margin_sweep_max))
        used_max = min(forced_max, observed_margin_max)
        if sweep_steps == 1:
            margin_vals = np.array([0.0], dtype=np.float64)
        else:
            margin_vals = np.linspace(0.0, used_max, sweep_steps, dtype=np.float64)
            margin_vals[0] = 0.0
    cluster_vals = adaptive_high_tail_sweep_values(
        df["cluster_confidence_p10"],
        sweep_steps=sweep_steps,
        q_lo=0.90,
        q_hi=0.999,
    )

    rows: list[dict[str, float | int]] = []
    total_queries = len(df)

    for m_t, c_t in zip(margin_vals, cluster_vals):
        filtered = apply_threshold(df, margin_t=m_t, cluster_t=c_t)
        retained = len(filtered)
        coverage = retained / total_queries if total_queries > 0 else 0.0

        row: dict[str, float | int] = {
            "margin_threshold": float(m_t),
            "cluster_confidence_threshold": float(c_t),
            "queries_retained": int(retained),
            "query_coverage": float(coverage),
        }

        if gt_available and not filtered.empty:
            precision, recall_retained, pred_pos, tp = compute_pairwise_pr_fast(filtered)
            recall_global = tp / full_true_pos_universe if full_true_pos_universe > 0 else float("nan")
            f1_retained = (
                2.0 * precision * recall_retained / (precision + recall_retained)
                if (precision + recall_retained) > 0
                else float("nan")
            )
            f1_global = (
                2.0 * precision * recall_global / (precision + recall_global)
                if (precision + recall_global) > 0
                else float("nan")
            )

            per_group = compute_groupwise_metrics(filtered)
            gp10, gp50, gp90 = percentile_triplet(per_group["pairwise_precision"])
            gr10, gr50, gr90 = percentile_triplet(per_group["pairwise_recall"])

            row["pairwise_precision"] = float(precision)
            row["pairwise_recall"] = float(recall_retained)
            row["pairwise_recall_global"] = float(recall_global)
            row["pairwise_f1"] = float(f1_retained)
            row["pairwise_f1_global"] = float(f1_global)
            row["pred_pair_positives"] = int(pred_pos)
            row["true_pair_positives"] = int(tp)
            row["group_precision_p10"] = gp10
            row["group_precision_p50"] = gp50
            row["group_precision_p90"] = gp90
            row["group_recall_p10"] = gr10
            row["group_recall_p50"] = gr50
            row["group_recall_p90"] = gr90
        else:
            row["pairwise_precision"] = float("nan")
            row["pairwise_recall"] = float("nan")
            row["pairwise_recall_global"] = float("nan")
            row["pairwise_f1"] = float("nan")
            row["pairwise_f1_global"] = float("nan")
            row["pred_pair_positives"] = 0
            row["true_pair_positives"] = 0
            row["group_precision_p10"] = float("nan")
            row["group_precision_p50"] = float("nan")
            row["group_precision_p90"] = float("nan")
            row["group_recall_p10"] = float("nan")
            row["group_recall_p50"] = float("nan")
            row["group_recall_p90"] = float("nan")

        rows.append(row)

    meta = {
        "margin_sweep_min": float(margin_vals.min()) if margin_vals.size else 0.0,
        "margin_sweep_max": float(margin_vals.max()) if margin_vals.size else 0.0,
        "margin_sweep_max_requested": float(margin_sweep_max) if margin_sweep_max is not None else None,
        "margin_sweep_max_observed": observed_margin_max,
        "cluster_sweep_min": float(cluster_vals.min()) if cluster_vals.size else 0.0,
        "cluster_sweep_max": float(cluster_vals.max()) if cluster_vals.size else 0.0,
    }
    return pd.DataFrame(rows), meta


def build_pair_examples(confident_df: pd.DataFrame, max_examples: int) -> pd.DataFrame:
    """Build example query pairs from confident assignments within predicted clusters."""
    if max_examples <= 0 or confident_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    has_truth = "AnonID_true" in confident_df.columns

    for (split, gid, cluster), grp in confident_df.groupby(
        ["split", "group_id", "predicted_cluster"], sort=False
    ):
        if len(grp) < 2:
            continue

        # Keep combinations bounded while preferring highest-margin rows.
        grp = grp.sort_values("assign_margin", ascending=False).head(25).reset_index(drop=True)

        for i, j in itertools.combinations(range(len(grp)), 2):
            a = grp.iloc[i]
            b = grp.iloc[j]

            rec: dict[str, object] = {
                "split": str(split),
                "group_id": int(gid),
                "predicted_cluster": int(cluster),
                "query_idx_a": int(a["query_idx"]),
                "query_idx_b": int(b["query_idx"]),
                "Query_a": str(a["Query"]),
                "Query_b": str(b["Query"]),
                "QueryTimeRounded24h_a": a["QueryTimeRounded24h"],
                "QueryTimeRounded24h_b": b["QueryTimeRounded24h"],
                "assign_margin_a": float(a["assign_margin"]),
                "assign_margin_b": float(b["assign_margin"]),
                "pair_mean_margin": float((float(a["assign_margin"]) + float(b["assign_margin"])) / 2.0),
                "cluster_confidence_p10": float(a["cluster_confidence_p10"]),
            }

            if has_truth:
                rec["AnonID_true_a"] = int(a["AnonID_true"])
                rec["AnonID_true_b"] = int(b["AnonID_true"])
                rec["same_true_user"] = int(int(a["AnonID_true"]) == int(b["AnonID_true"]))

            rows.append(rec)

            if len(rows) >= max_examples * 10:
                break

        if len(rows) >= max_examples * 10:
            break

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("pair_mean_margin", ascending=False).head(max_examples)
    return out.reset_index(drop=True)


def main() -> None:
    args = parse_args()

    print(f"Loading assignments: {args.assignments_path}")
    df = load_assignments(args.assignments_path)
    gt_available = has_ground_truth(df)
    print(f"Total queries: {len(df):,}")
    print(f"Ground truth available: {gt_available}")

    # Apply user-specified threshold and report.
    confident = apply_threshold(
        df,
        margin_t=args.margin_threshold,
        cluster_t=args.cluster_confidence_threshold,
    )
    retained = len(confident)
    coverage = retained / len(df) if len(df) > 0 else 0.0
    print(
        f"Retained at margin>={args.margin_threshold:.2f}, "
        f"cluster_conf>={args.cluster_confidence_threshold:.2f}: "
        f"{retained:,} queries ({coverage:.1%} coverage)"
    )

    result: dict[str, float | int | None] = {
        "margin_threshold": float(args.margin_threshold),
        "cluster_confidence_threshold": float(args.cluster_confidence_threshold),
        "total_queries": int(len(df)),
        "queries_retained": int(retained),
        "query_coverage": float(coverage),
        "pairwise_precision": None,
        "pairwise_recall": None,
        "pairwise_recall_global": None,
        "pairwise_f1": None,
        "pairwise_f1_global": None,
        "group_precision_p10": None,
        "group_precision_p50": None,
        "group_precision_p90": None,
        "group_recall_p10": None,
        "group_recall_p50": None,
        "group_recall_p90": None,
    }

    full_true_pos_universe = 0
    if gt_available:
        _, _, full_true_pos_universe = compute_pairwise_counts(df)

    if gt_available and not confident.empty:
        precision, recall_retained, pred_pos, tp = compute_pairwise_pr_fast(confident)
        recall_global = tp / full_true_pos_universe if full_true_pos_universe > 0 else float("nan")
        f1_retained = (
            2.0 * precision * recall_retained / (precision + recall_retained)
            if (precision + recall_retained) > 0
            else float("nan")
        )
        f1_global = (
            2.0 * precision * recall_global / (precision + recall_global)
            if (precision + recall_global) > 0
            else float("nan")
        )
        per_group = compute_groupwise_metrics(confident)
        gp10, gp50, gp90 = percentile_triplet(per_group["pairwise_precision"])
        gr10, gr50, gr90 = percentile_triplet(per_group["pairwise_recall"])

        result["pairwise_precision"] = float(precision)
        result["pairwise_recall"] = float(recall_retained)
        result["pairwise_recall_global"] = float(recall_global)
        result["pairwise_f1"] = float(f1_retained)
        result["pairwise_f1_global"] = float(f1_global)
        result["pred_pair_positives"] = int(pred_pos)
        result["true_pair_positives"] = int(tp)
        result["group_precision_p10"] = gp10
        result["group_precision_p50"] = gp50
        result["group_precision_p90"] = gp90
        result["group_recall_p10"] = gr10
        result["group_recall_p50"] = gr50
        result["group_recall_p90"] = gr90

        print(f"Pairwise precision: {precision:.4f}")
        print(f"Pairwise recall (retained): {recall_retained:.4f}")
        print(f"Pairwise recall (global):   {recall_global:.4f}")
        print(f"Pairwise F1 (retained):     {f1_retained:.4f}")
        print(f"Pairwise F1 (global):       {f1_global:.4f}")
        print("Per-group precision p10/p50/p90:", f"{gp10:.4f}/{gp50:.4f}/{gp90:.4f}")
        print("Per-group recall    p10/p50/p90:", f"{gr10:.4f}/{gr50:.4f}/{gr90:.4f}")

    # Threshold sweep.
    print(f"Running threshold sweep ({args.sweep_steps} steps)...")
    sweep_df, sweep_meta = build_threshold_sweep(
        df,
        sweep_steps=args.sweep_steps,
        gt_available=gt_available,
        full_true_pos_universe=full_true_pos_universe,
        margin_sweep_max=args.margin_sweep_max,
    )
    print(
        "Adaptive sweep ranges: "
        f"margin [{sweep_meta['margin_sweep_min']:.4f}, {sweep_meta['margin_sweep_max']:.4f}], "
        f"cluster_conf [{sweep_meta['cluster_sweep_min']:.4f}, {sweep_meta['cluster_sweep_max']:.4f}]"
    )
    if sweep_meta["margin_sweep_max_requested"] is not None:
        print(
            "Margin sweep max requested: "
            f"{sweep_meta['margin_sweep_max_requested']:.4f} "
            f"(observed max: {sweep_meta['margin_sweep_max_observed']:.4f})"
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    confident_path = out_dir / "confident_assignments.parquet"
    sweep_parquet_path = out_dir / "confident_threshold_sweep.parquet"
    sweep_json_path = out_dir / "confident_threshold_sweep.json"
    result_json_path = out_dir / "confident_result.json"
    pair_examples_path = out_dir / "confident_pair_examples.parquet"

    confident.to_parquet(confident_path, index=False)
    sweep_df.to_parquet(sweep_parquet_path, index=False)

    # Replace NaN with null-safe representation for JSON.
    sweep_records = sweep_df.where(pd.notnull(sweep_df), other=None).to_dict(orient="records")
    sweep_json_path.write_text(json.dumps(sweep_records, indent=2), encoding="utf-8")
    result_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Wrote: {confident_path}")
    print(f"Wrote: {sweep_parquet_path}")
    print(f"Wrote: {sweep_json_path}")
    print(f"Wrote: {result_json_path}")

    # Optional pair example extraction for manual inspection.
    if args.print_pair_examples > 0:
        pair_examples = build_pair_examples(confident, max_examples=args.print_pair_examples)
        if pair_examples.empty:
            print("No confident pair examples available at current thresholds.")
        else:
            pair_examples.to_parquet(pair_examples_path, index=False)
            print(f"Wrote: {pair_examples_path}")
            print("\nExample confident pairs:")
            display_cols = [
                "split",
                "group_id",
                "predicted_cluster",
                "pair_mean_margin",
                "Query_a",
                "Query_b",
            ]
            if "same_true_user" in pair_examples.columns:
                display_cols.append("same_true_user")
            print(pair_examples[display_cols].to_string(index=False, max_colwidth=70))

    # Print sweep summary.
    print("\nThreshold sweep summary:")
    display_cols = [
        "margin_threshold",
        "cluster_confidence_threshold",
        "query_coverage",
    ]
    if gt_available:
        display_cols += [
            "pairwise_precision",
            "pairwise_recall",
            "pairwise_recall_global",
            "pairwise_f1",
            "pairwise_f1_global",
        ]
    print(sweep_df[display_cols].to_string(index=False, float_format="{:.6f}".format))


if __name__ == "__main__":
    main()
