"""
Direct pairwise threshold evaluation for the 50-user linkage setting.

Instead of clustering, this script thresholds calibrated pairwise same-user
probabilities directly and measures precision/recall/F1 across a sweep of
thresholds.

Steps:
1. Load sampled groups from group_manifest.parquet.
2. For each group, score all query pairs with linkage_pair_model.pkl.
3. Checkpoint scored pairs to disk (pairs above --min-save-score).
4. Sweep thresholds and compute precision/recall/F1/pair-coverage.
5. Print and save top-N highest-scoring pairs.

Outputs:
- pair_eval_sweep.parquet / pair_eval_sweep.json: threshold sweep table
- pair_eval_top_pairs.parquet: top-N pairs by score with query text
- pair_eval_result.json: metrics at the chosen (or best-F1) threshold
- groups/: per-group checkpoint files (scored pairs + top pairs + meta)
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_PATH = BASE_DIR / "raw.parquet"
DEFAULT_GROUP_MANIFEST_PATH = BASE_DIR / "linkage_work" / "group_manifest.parquet"
DEFAULT_MODEL_PATH = BASE_DIR / "linkage_work" / "model" / "linkage_pair_model.pkl"
DEFAULT_OUT_DIR = BASE_DIR / "linkage_work" / "pair_eval"

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_PAIR_BATCH = 500_000
_TOP_K_PER_GROUP = 500  # top pairs saved per group checkpoint for global top-N


def apply_calibrator(calibrator, base_proba: np.ndarray) -> np.ndarray:
    # Support new sigmoid calibrator (predict_proba) and legacy isotonic (predict).
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(base_proba.reshape(-1, 1))[:, 1]
    if hasattr(calibrator, "predict"):
        return calibrator.predict(base_proba)
    raise TypeError("Unsupported calibrator object: missing predict/predict_proba")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct pairwise threshold evaluation (no clustering)."
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
        help="Which manifest split to run (default: val).",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="Cap on number of groups to process (0 = all).",
    )
    parser.add_argument(
        "--max-queries-per-user",
        type=int,
        default=100,
        help="Cap per-user queries per group for compute control (default: 100).",
    )
    parser.add_argument(
        "--min-save-score",
        type=float,
        default=0.3,
        help=(
            "Only checkpoint pairs with score >= this value (default: 0.3). "
            "Lower values increase disk usage but improve low-threshold recall estimates."
        ),
    )
    parser.add_argument(
        "--sweep-steps",
        type=int,
        default=30,
        help=(
            "Sweep display/control parameter (default: 30). "
            "In threshold-geometric mode, controls how many highest-threshold rows are printed. "
            "In recall-halving mode, caps the number of halving targets evaluated."
        ),
    )
    parser.add_argument(
        "--sweep-mode",
        default="threshold-geometric",
        choices=["threshold-geometric", "recall-halving"],
        help=(
            "Sweep strategy. threshold-geometric: threshold starts at max(0.3,min-save) "
            "and moves 10% toward 1.0 until NULL. recall-halving: target recall levels "
            "1.0, 0.5, 0.25, ... down to 0."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Score threshold for the final result output. "
            "If omitted, the threshold with the best F1 is chosen automatically."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of highest-scoring pairs to print and save (default: 20).",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip scoring; merge existing checkpoints and compute sweep.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model + data helpers (shared with linkage_infer_groups.py)
# ---------------------------------------------------------------------------

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


def tokenize(text: str) -> set[str]:
    return {tok for tok in TOKEN_RE.findall(text.lower()) if tok}


def extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""


def _pa_array_int64(values: np.ndarray):
    import pyarrow as pa
    return pa.array(values, type=pa.int64())


def load_manifest(path: Path, split: str, max_groups: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Group manifest not found: {path}")
    manifest = pd.read_parquet(path)
    if split != "all":
        manifest = manifest[manifest["split"] == split].copy()
    manifest = manifest.sort_values(["split", "group_id", "AnonID"]).reset_index(drop=True)
    if max_groups > 0:
        keep = manifest[["split", "group_id"]].drop_duplicates().head(max_groups)
        manifest = manifest.merge(keep, on=["split", "group_id"], how="inner")
    if manifest.empty:
        raise ValueError("Manifest is empty. Check --split and --max-groups.")
    return manifest


def load_group_queries(raw_path: Path, group_users: np.ndarray) -> pd.DataFrame:
    dataset = ds.dataset(raw_path.as_posix(), format="parquet")
    filter_expr = pc.is_in(ds.field("AnonID"), value_set=_pa_array_int64(group_users))
    table = dataset.to_table(
        columns=["AnonID", "Query", "QueryTime", "ClickURL"],
        filter=filter_expr,
    )
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
    out = df.copy()
    out["_rank"] = out.groupby("AnonID").cumcount()
    out = out[out["_rank"] < cap].drop(columns=["_rank"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Pair scoring
# ---------------------------------------------------------------------------

def score_all_pairs(
    feature_columns: list[str],
    model,
    calibrator,
    anon_ids: np.ndarray,
    queries: np.ndarray,
    query_times: np.ndarray,
    click_urls: np.ndarray,
    min_save_score: float,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Score every upper-triangle pair in the group.

    Returns:
      scored_df  — pairs with score >= min_save_score; columns: score, same_user_true
      top_df     — top_k pairs with full info for display
      total_pairs     — total pairs considered (for coverage denominator)
      total_positives — total same-user pairs (for recall denominator)
    """
    n = len(queries)
    if n <= 1:
        empty_scored = pd.DataFrame({"score": pd.Series(dtype="float32"),
                                     "same_user_true": pd.Series(dtype=bool)})
        return empty_scored, pd.DataFrame(), 0, 0

    i_idx, j_idx = np.triu_indices(n, k=1)
    n_pairs = len(i_idx)

    same_user_true_all = anon_ids[i_idx] == anon_ids[j_idx]
    total_positives = int(same_user_true_all.sum())

    # Pre-compute per-query derived values once
    str_queries = [str(q) for q in queries]
    token_sets = [tokenize(q) for q in str_queries]
    str_urls = [str(u) for u in click_urls]
    domains = [extract_domain(u) for u in str_urls]
    prefix3 = [q[:3].lower() for q in str_queries]
    has_digit = np.array([int(any(ch.isdigit() for ch in q)) for q in str_queries])
    query_lower = [q.strip().lower() for q in str_queries]
    has_url = np.array([int(bool(u)) for u in str_urls])

    calibrated_all = np.empty(n_pairs, dtype=np.float32)

    for start in range(0, n_pairs, _PAIR_BATCH):
        end = min(start + _PAIR_BATCH, n_pairs)
        ia = i_idx[start:end]
        ja = j_idx[start:end]
        m = end - start

        day_gap = np.abs(
            (query_times[ia] - query_times[ja]) / np.timedelta64(1, "D")
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
            "token_jaccard": np.divide(
                inter, union, out=np.zeros(m, dtype=np.float64), where=union > 0
            ),
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
        calibrated_all[start:end] = apply_calibrator(calibrator, base).astype(np.float32)

    # Pairs above min_save_score (for sweep)
    keep_mask = calibrated_all >= min_save_score
    scored_df = pd.DataFrame({
        "score": calibrated_all[keep_mask],
        "same_user_true": same_user_true_all[keep_mask],
    })

    # Top-K pairs with full info (for display)
    actual_top_k = min(top_k, n_pairs)
    top_idx_unsorted = np.argpartition(calibrated_all, -actual_top_k)[-actual_top_k:]
    top_idx = top_idx_unsorted[np.argsort(-calibrated_all[top_idx_unsorted])]
    top_ia = i_idx[top_idx]
    top_ja = j_idx[top_idx]

    top_df = pd.DataFrame({
        "score": calibrated_all[top_idx],
        "same_user_true": same_user_true_all[top_idx],
        "anon_id_a": anon_ids[top_ia],
        "anon_id_b": anon_ids[top_ja],
        "Query_a": [str_queries[k] for k in top_ia],
        "Query_b": [str_queries[k] for k in top_ja],
        "QueryTime_a": query_times[top_ia],
        "QueryTime_b": query_times[top_ja],
        "ClickURL_a": [str_urls[k] for k in top_ia],
        "ClickURL_b": [str_urls[k] for k in top_ja],
    })

    return scored_df, top_df, int(n_pairs), total_positives


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _stem(split_name: str, group_id: int) -> str:
    return f"{split_name}_{group_id:06d}"


def _checkpoint_exists(groups_dir: Path, split_name: str, group_id: int) -> bool:
    stem = _stem(split_name, group_id)
    if not (groups_dir / f"{stem}_meta.json").exists():
        return False
    top_path = groups_dir / f"{stem}_top.parquet"
    if not top_path.exists():
        return False
    try:
        cols = pd.read_parquet(top_path, columns=[]).columns.tolist()
        if "ClickURL_a" not in cols or "ClickURL_b" not in cols:
            return False
    except Exception:
        return False
    return True


def _write_checkpoint(
    groups_dir: Path,
    split_name: str,
    group_id: int,
    scored_df: pd.DataFrame,
    top_df: pd.DataFrame,
    total_pairs: int,
    total_positives: int,
) -> None:
    stem = _stem(split_name, group_id)
    scored_df.to_parquet(groups_dir / f"{stem}_scored.parquet", index=False)
    top_df.to_parquet(groups_dir / f"{stem}_top.parquet", index=False)
    meta = {
        "total_pairs": total_pairs,
        "total_positives": total_positives,
        "split": split_name,
        "group_id": group_id,
    }
    (groups_dir / f"{stem}_meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Sweep + output
# ---------------------------------------------------------------------------

def compute_sweep(
    all_scores: np.ndarray,
    all_labels: np.ndarray,
    total_positives: int,
    total_pairs: int,
    min_save_score: float,
    sweep_mode: str,
    sweep_steps: int,
) -> pd.DataFrame:
    if len(all_scores) == 0:
        return pd.DataFrame(
            columns=["threshold", "n_predicted", "true_positives", "false_positives",
                     "precision", "recall", "f1", "pair_coverage"]
        )

    def row_for_threshold(t: float) -> dict[str, float | int]:
        mask = all_scores >= t
        n_pred = int(mask.sum())
        tp = int(all_labels[mask].sum())
        fp = n_pred - tp
        prec = tp / n_pred if n_pred > 0 else float("nan")
        recall = tp / total_positives if total_positives > 0 else float("nan")
        f1 = (
            2 * prec * recall / (prec + recall)
            if (not np.isnan(prec) and not np.isnan(recall) and (prec + recall) > 0)
            else float("nan")
        )
        coverage = n_pred / total_pairs if total_pairs > 0 else float("nan")
        return {
            "threshold": float(t),
            "n_predicted": n_pred,
            "true_positives": tp,
            "false_positives": fp,
            "precision": prec,
            "recall": recall,
            "f1": f1,
            "pair_coverage": coverage,
        }

    rows = []

    if sweep_mode == "recall-halving":
        # Recall targets: 1.0, 0.5, 0.25, ... then terminal 0.0.
        # Note: achieving 1.0 requires saving all true positives (e.g., --min-save-score 0.0).
        pos_scores = all_scores[all_labels]
        if total_positives <= 0 or pos_scores.size == 0:
            rows.append(row_for_threshold(float(np.nextafter(1.0, 2.0))))
            return pd.DataFrame(rows)

        max_targets = max(1, int(sweep_steps))
        target = 1.0
        targets: list[float] = [1.0]
        while len(targets) < max_targets:
            target *= 0.5
            if target <= 0.0:
                break
            targets.append(float(target))
        if targets[-1] != 0.0:
            targets.append(0.0)

        seen_thresholds: set[float] = set()
        for r_target in targets:
            if r_target <= 0.0:
                t = float(np.nextafter(1.0, 2.0))
            else:
                q = max(0.0, min(1.0, 1.0 - r_target))
                t = float(np.quantile(pos_scores, q, method="higher"))
                t = max(t, float(min_save_score))

            if t in seen_thresholds:
                continue
            seen_thresholds.add(t)
            rows.append(row_for_threshold(t))

        if rows[-1]["n_predicted"] != 0:
            rows.append(row_for_threshold(float(np.nextafter(1.0, 2.0))))

        rows = sorted(rows, key=lambda d: float(d["threshold"]))
        return pd.DataFrame(rows)

    # threshold-geometric mode
    # Start at 0.3, then increase by 10% of remaining distance to 1.0.
    # If min_save_score is above 0.3, start from min_save_score so counts remain valid.
    t = max(0.3, float(min_save_score))
    max_iters = 10000
    for _ in range(max_iters):
        row = row_for_threshold(t)
        rows.append(row)

        # Stop once the sweep reaches a NULL prediction row.
        if row["n_predicted"] == 0:
            break

        next_t = t + 0.1 * (1.0 - t)
        if next_t <= t:
            # Numerical safeguard: force a threshold above 1.0 to guarantee NULL.
            t = float(np.nextafter(1.0, 2.0))
        else:
            t = next_t

    if rows and rows[-1]["n_predicted"] != 0:
        # Ensure a terminal NULL row exists even in edge cases.
        rows.append(row_for_threshold(float(np.nextafter(1.0, 2.0))))

    return pd.DataFrame(rows)


def merge_and_evaluate(
    groups_dir: Path,
    out_dir: Path,
    n_steps: int,
    sweep_mode: str,
    top_n: int,
    threshold: float | None,
    min_save_score: float,
) -> None:
    meta_files = sorted(groups_dir.glob("*_meta.json"))
    if not meta_files:
        print("No checkpoint files found to merge.")
        return

    scored_parts: list[pd.DataFrame] = []
    top_parts: list[pd.DataFrame] = []
    total_positives = 0
    total_pairs = 0

    for mf in meta_files:
        meta = json.loads(mf.read_text(encoding="utf-8"))
        total_positives += meta["total_positives"]
        total_pairs += meta["total_pairs"]
        stem = mf.stem.replace("_meta", "")
        sp = groups_dir / f"{stem}_scored.parquet"
        tp = groups_dir / f"{stem}_top.parquet"
        if sp.exists():
            scored_parts.append(pd.read_parquet(sp))
        if tp.exists():
            top_parts.append(pd.read_parquet(tp))

    print(
        f"Merged {len(meta_files)} group(s): "
        f"{total_pairs:,} total pairs, {total_positives:,} same-user pairs."
    )

    all_scores = (
        np.concatenate([df["score"].to_numpy(np.float32) for df in scored_parts])
        if scored_parts else np.array([], dtype=np.float32)
    )
    all_labels = (
        np.concatenate([df["same_user_true"].to_numpy(bool) for df in scored_parts])
        if scored_parts else np.array([], dtype=bool)
    )

    sweep = compute_sweep(
        all_scores,
        all_labels,
        total_positives,
        total_pairs,
        min_save_score,
        sweep_mode,
        n_steps,
    )
    sweep_path = out_dir / "pair_eval_sweep.parquet"
    sweep_json_path = out_dir / "pair_eval_sweep.json"
    sweep.to_parquet(sweep_path, index=False)
    sweep_json_path.write_text(
        sweep.to_json(orient="records", indent=2), encoding="utf-8"
    )
    print(f"Wrote sweep: {sweep_path}")

    # Choose operating threshold
    if threshold is None:
        valid = sweep.dropna(subset=["f1"])
        best_row = valid.loc[valid["f1"].idxmax()] if not valid.empty else sweep.iloc[0]
        threshold = float(best_row["threshold"])
        print(f"Best-F1 threshold chosen automatically: {threshold:.6f}")

    chosen = sweep[sweep["threshold"] >= threshold]
    result_dict = (chosen.iloc[0] if not chosen.empty else sweep.iloc[-1]).to_dict()
    result_dict["threshold_used"] = threshold
    result_path = out_dir / "pair_eval_result.json"
    result_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
    p = result_dict.get("precision", float("nan"))
    r = result_dict.get("recall", float("nan"))
    f = result_dict.get("f1", float("nan"))
    cov = result_dict.get("pair_coverage", float("nan"))
    print(
        f"Result at threshold {threshold:.6f}: "
        f"precision={p:.6f}  recall={r:.6f}  f1={f:.6f}  pair_coverage={cov:.6f}"
    )
    print(f"Wrote result: {result_path}")

    # Top-N pairs
    top_all = None
    if top_parts:
        top_all = (
            pd.concat(top_parts, ignore_index=True)
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
        top_path = out_dir / "pair_eval_top_pairs.parquet"
        top_all.head(top_n).to_parquet(top_path, index=False)
        print(f"\nTop {top_n} pairs:")
        for _, row in top_all.head(top_n).iterrows():
            label = "SAME" if row.get("same_user_true", False) else "DIFF"
            ts_a = str(row.get("QueryTime_a", ""))[:19]
            ts_b = str(row.get("QueryTime_b", ""))[:19]
            url_a = row.get("ClickURL_a", "") or "(no ClickURL)"
            url_b = row.get("ClickURL_b", "") or "(no ClickURL)"
            print(f"  score={row['score']:.6f} [{label}]")
            print(f"    A: {str(row['Query_a'])!r}  {ts_a}  {url_a}")
            print(f"    B: {str(row['Query_b'])!r}  {ts_b}  {url_b}")
        print(f"Wrote top pairs: {top_path}")

        # Top-10 pairs with identical Query, same-day QueryTime, and same (or absent) ClickURL
        if "ClickURL_a" not in top_all.columns or "ClickURL_b" not in top_all.columns:
            print("\n(Skipping identical-query filter: ClickURL columns absent in checkpoints — re-score to populate.)")
        else:
            ta = pd.to_datetime(top_all["QueryTime_a"], errors="coerce").dt.normalize()
            tb = pd.to_datetime(top_all["QueryTime_b"], errors="coerce").dt.normalize()
            url_a_col = top_all["ClickURL_a"].fillna("").astype(str).str.strip()
            url_b_col = top_all["ClickURL_b"].fillna("").astype(str).str.strip()
            query_match = (
                top_all["Query_a"].astype(str).str.strip().str.lower()
                == top_all["Query_b"].astype(str).str.strip().str.lower()
            )
            time_match = ta == tb
            url_match = (url_a_col == url_b_col) | (url_a_col == "") | (url_b_col == "")
            identical = top_all[query_match & time_match & url_match].head(10)
            if not identical.empty:
                print(f"\nTop {len(identical)} pairs with identical Query / same-day time / same-or-null ClickURL:")
                for _, row in identical.iterrows():
                    label = "SAME" if row.get("same_user_true", False) else "DIFF"
                    ts_a = str(row.get("QueryTime_a", ""))[:19]
                    ts_b = str(row.get("QueryTime_b", ""))[:19]
                    url_a = row.get("ClickURL_a", "") or "(no ClickURL)"
                    url_b = row.get("ClickURL_b", "") or "(no ClickURL)"
                    print(f"  score={row['score']:.6f} [{label}]")
                    print(f"    A: {str(row['Query_a'])!r}  {ts_a}  {url_a}")
                    print(f"    B: {str(row['Query_b'])!r}  {ts_b}  {url_b}")
            else:
                print("\nNo pairs with identical Query / same-day time / same-or-null ClickURL found in checkpoints.")

    if sweep_mode == "recall-halving":
        print("\nSweep (recall-halving targets):")
        print(sweep.to_string(index=False))
    else:
        print(f"\nSweep (highest {n_steps} thresholds):")
        print(sweep.tail(max(1, int(n_steps))).to_string(index=False))

    if top_parts:
        max_score = float(top_all["score"].iloc[0])
        mask = all_scores >= max_score
        n_pred = int(mask.sum())
        tp = int(all_labels[mask].sum())
        fp = n_pred - tp
        prec = tp / n_pred if n_pred > 0 else float("nan")
        rec = tp / total_positives if total_positives > 0 else float("nan")
        f1_at_max = (
            2 * prec * rec / (prec + rec)
            if (not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0)
            else float("nan")
        )
        cov_at_max = n_pred / total_pairs if total_pairs > 0 else float("nan")
        print(f"\nAt max score ({max_score:.6f}):")
        print(
            f"  n_predicted={n_pred}  true_positives={tp}  false_positives={fp}  "
            f"precision={prec:.6f}  recall={rec:.6f}  f1={f1_at_max:.6f}  "
            f"pair_coverage={cov_at_max:.6f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = args.out_dir / "groups"
    groups_dir.mkdir(exist_ok=True)

    if args.merge_only:
        print("--merge-only: merging existing checkpoints.")
        merge_and_evaluate(
            groups_dir, args.out_dir,
            args.sweep_steps,
            args.sweep_mode,
            args.top_n,
            args.threshold,
            args.min_save_score,
        )
        return

    if not args.raw_path.exists():
        raise FileNotFoundError(f"raw.parquet not found: {args.raw_path}")

    artifact = load_model(args.model_path)
    model = artifact["model"]
    calibrator = artifact["calibrator"]
    feature_columns = artifact["feature_columns"]

    manifest = load_manifest(args.group_manifest_path, args.split, args.max_groups)
    group_keys = (
        manifest[["split", "group_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    n_total = len(group_keys)
    n_done = 0

    print(
        f"Processing groups: {n_total:,} "
        f"(checkpoints in: {groups_dir})"
    )

    for _, row in group_keys.iterrows():
        split_name = str(row["split"])
        group_id = int(row["group_id"])

        if _checkpoint_exists(groups_dir, split_name, group_id):
            n_done += 1
            continue

        group_users = manifest[
            (manifest["split"] == split_name) & (manifest["group_id"] == group_id)
        ]["AnonID"].unique()

        df = load_group_queries(args.raw_path, group_users)
        df = cap_queries_per_user(df, args.max_queries_per_user)

        if df.empty:
            print(f"  Group {group_id}: no queries after capping, skipping.")
            continue

        anon_ids = df["AnonID"].to_numpy(np.int64)
        queries = df["Query"].to_numpy()
        query_times = df["QueryTime"].to_numpy()
        click_urls = df["ClickURL"].fillna("").to_numpy()

        scored_df, top_df, total_pairs, total_positives = score_all_pairs(
            feature_columns=feature_columns,
            model=model,
            calibrator=calibrator,
            anon_ids=anon_ids,
            queries=queries,
            query_times=query_times,
            click_urls=click_urls,
            min_save_score=args.min_save_score,
            top_k=_TOP_K_PER_GROUP,
        )

        _write_checkpoint(
            groups_dir, split_name, group_id,
            scored_df, top_df, total_pairs, total_positives,
        )

        n_done += 1
        print(
            f"  {n_done}/{n_total}  group={group_id}  "
            f"pairs={total_pairs:,}  positives={total_positives:,}  "
            f"saved={len(scored_df):,}"
        )

    merge_and_evaluate(
        groups_dir, args.out_dir,
        args.sweep_steps,
        args.sweep_mode,
        args.top_n,
        args.threshold,
        args.min_save_score,
    )


if __name__ == "__main__":
    main()
