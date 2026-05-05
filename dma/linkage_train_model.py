"""
Train and calibrate a pairwise same-user probability model.

Input:
- linkage_work/train_pairs.parquet from linkage_build_pairs.py

Outputs (under --out-dir):
- linkage_pair_model.pkl: serialized base model + calibrator + feature list
- linkage_pair_metrics.json: holdout metrics summary
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PAIRS_PATH = BASE_DIR / "linkage_work" / "train_pairs.parquet"
DEFAULT_OUT_DIR = BASE_DIR / "linkage_work" / "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and calibrate a pairwise same-user probability model."
    )
    parser.add_argument(
        "--pairs-path",
        type=Path,
        default=DEFAULT_PAIRS_PATH,
        help=f"Path to train_pairs.parquet (default: {DEFAULT_PAIRS_PATH}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--calib-user-fraction",
        type=float,
        default=0.2,
        help="Fraction of users reserved for calibration holdout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Max iterations for HistGradientBoostingClassifier.",
    )
    return parser.parse_args()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"row_id_a", "row_id_b", "label", "anon_id_a", "anon_id_b"}
    features = [c for c in df.columns if c not in skip]
    if not features:
        raise ValueError("No feature columns found in pair file.")
    return features


def user_based_split(
    df: pd.DataFrame,
    calib_user_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if calib_user_fraction <= 0 or calib_user_fraction >= 1:
        raise ValueError("--calib-user-fraction must be in (0, 1).")

    users = pd.unique(pd.concat([df["anon_id_a"], df["anon_id_b"]], ignore_index=True))
    users = pd.Series(users).dropna().astype("int64").to_numpy()

    rng = np.random.default_rng(seed)
    shuffled = users.copy()
    rng.shuffle(shuffled)

    split_idx = int((1.0 - calib_user_fraction) * shuffled.size)
    train_users = set(shuffled[:split_idx].tolist())
    calib_users = set(shuffled[split_idx:].tolist())

    mask_train = df["anon_id_a"].isin(train_users) & df["anon_id_b"].isin(train_users)
    mask_calib = df["anon_id_a"].isin(calib_users) & df["anon_id_b"].isin(calib_users)

    train_df = df.loc[mask_train].copy()
    calib_df = df.loc[mask_calib].copy()

    # Fallback if user-based split is too small or single-class due to pair structure.
    if (
        train_df.empty
        or calib_df.empty
        or train_df["label"].nunique() < 2
        or calib_df["label"].nunique() < 2
    ):
        train_df, calib_df = train_test_split(
            df,
            test_size=calib_user_fraction,
            random_state=seed,
            stratify=df["label"],
        )
        train_df = train_df.copy()
        calib_df = calib_df.copy()

    return train_df, calib_df


def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def main() -> None:
    args = parse_args()

    if not args.pairs_path.exists():
        raise FileNotFoundError(f"pairs file not found: {args.pairs_path}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pairs from: {args.pairs_path}")
    df = pd.read_parquet(args.pairs_path)

    required = {"label", "anon_id_a", "anon_id_b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in pair file: {sorted(missing)}")

    feature_columns = get_feature_columns(df)

    train_df, calib_df = user_based_split(
        df=df,
        calib_user_fraction=args.calib_user_fraction,
        seed=args.seed,
    )

    print(f"Train rows: {len(train_df):,}")
    print(f"Calib rows: {len(calib_df):,}")

    x_train = train_df[feature_columns].to_numpy(dtype=np.float64)
    y_train = train_df["label"].to_numpy(dtype=np.int64)

    x_calib = calib_df[feature_columns].to_numpy(dtype=np.float64)
    y_calib = calib_df["label"].to_numpy(dtype=np.int64)

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=args.max_iter,
        max_depth=8,
        min_samples_leaf=50,
        random_state=args.seed,
    )
    model.fit(x_train, y_train)

    base_calib = model.predict_proba(x_calib)[:, 1]

    # Use sigmoid (Platt) calibration to avoid large flat regions at exactly 0/1.
    calibrator = LogisticRegression(random_state=args.seed)
    calibrator.fit(base_calib.reshape(-1, 1), y_calib)
    calibrated_calib = calibrator.predict_proba(base_calib.reshape(-1, 1))[:, 1]

    base_metrics = compute_metrics(y_calib, base_calib)
    calibrated_metrics = compute_metrics(y_calib, calibrated_calib)

    artifact = {
        "model": model,
        "calibrator": calibrator,
        "feature_columns": feature_columns,
        "metadata": {
            "seed": int(args.seed),
            "pairs_path": str(args.pairs_path),
            "train_rows": int(len(train_df)),
            "calib_rows": int(len(calib_df)),
            "calib_user_fraction": float(args.calib_user_fraction),
            "max_iter": int(args.max_iter),
            "calibrator_type": "sigmoid_platt_logistic",
        },
    }

    model_path = out_dir / "linkage_pair_model.pkl"
    metrics_path = out_dir / "linkage_pair_metrics.json"

    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    metrics_payload = {
        "base": base_metrics,
        "calibrated": calibrated_metrics,
        "label_rate_calib": float(y_calib.mean()),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Wrote: {model_path}")
    print(f"Wrote: {metrics_path}")
    print("Calibration metrics:")
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
