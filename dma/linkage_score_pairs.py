"""
Score pair rows with a trained linkage model and calibrated probabilities.

Input:
- pair feature parquet with columns matching training features
- linkage_pair_model.pkl from linkage_train_model.py

Output:
- parquet with row ids, optional labels, and probabilities
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PAIRS_PATH = BASE_DIR / "linkage_work" / "train_pairs.parquet"
DEFAULT_MODEL_PATH = BASE_DIR / "linkage_work" / "linkage_pair_model.pkl"
DEFAULT_OUTPUT_PATH = BASE_DIR / "linkage_work" / "train_pairs_scored.parquet"


def apply_calibrator(calibrator, base_proba: np.ndarray) -> np.ndarray:
    # Support new sigmoid calibrator (predict_proba) and legacy isotonic (predict).
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(base_proba.reshape(-1, 1))[:, 1]
    if hasattr(calibrator, "predict"):
        return calibrator.predict(base_proba)
    raise TypeError("Unsupported calibrator object: missing predict/predict_proba")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score pair rows and output calibrated same-user probabilities."
    )
    parser.add_argument(
        "--pairs-path",
        type=Path,
        default=DEFAULT_PAIRS_PATH,
        help=f"Input pair parquet (default: {DEFAULT_PAIRS_PATH}).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Model artifact path (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output parquet path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pairs_path.exists():
        raise FileNotFoundError(f"pairs file not found: {args.pairs_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"model artifact not found: {args.model_path}")

    print(f"Loading model: {args.model_path}")
    with args.model_path.open("rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    calibrator = artifact["calibrator"]
    feature_columns = artifact["feature_columns"]

    print(f"Loading pairs: {args.pairs_path}")
    df = pd.read_parquet(args.pairs_path)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Input pairs missing required feature columns: {missing}")

    x = df[feature_columns].to_numpy(dtype=np.float64)

    base = model.predict_proba(x)[:, 1]
    calibrated = apply_calibrator(calibrator, base)

    keep_cols = [c for c in ["row_id_a", "row_id_b", "label", "anon_id_a", "anon_id_b"] if c in df.columns]
    out = df[keep_cols].copy()
    out["proba_base"] = base
    out["proba_calibrated"] = calibrated

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_path, index=False)

    print(f"Wrote: {args.output_path}")
    print(
        "Calibrated probability summary: "
        f"min={out['proba_calibrated'].min():.4f}, "
        f"median={out['proba_calibrated'].median():.4f}, "
        f"max={out['proba_calibrated'].max():.4f}"
    )


if __name__ == "__main__":
    main()
