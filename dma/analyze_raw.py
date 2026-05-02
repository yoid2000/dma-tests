from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_PATH = BASE_DIR / "raw.parquet"
DEFAULT_PLOTS_DIR = BASE_DIR / "plots"


def run_query_length(raw_path: Path, plots_dir: Path) -> None:
    """Plot CDF ordering of query lengths (short to long) with length on y-axis."""
    print("Running analysis: query_length")
    df = pd.read_parquet(raw_path, columns=["Query"])

    query_lengths = df["Query"].astype("string").str.len().dropna().astype("int32")
    counts = query_lengths.value_counts().sort_index()
    cumulative = counts.cumsum()

    x_length = counts.index.to_numpy()
    y_cdf = (cumulative / cumulative.iloc[-1]).to_numpy()

    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "query_length_cdf.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(x_length, y_cdf, where="post", linewidth=2, color="#1f77b4")
    ax.axvline(100, color="#c62828", linestyle="--", linewidth=1.8, label="100 characters")
    ax.set_title("CDF Ordering of Query Lengths")
    ax.set_xlabel("Query Length (characters)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Wrote plot: {out_path}")


ANALYSES = {
    "query_length": run_query_length,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run analyses on dma/raw.parquet. Pass one analysis keyword to run a "
            "single analysis. If omitted, all analyses are run."
        )
    )
    parser.add_argument(
        "keyword",
        nargs="?",
        help="Analysis keyword to run (for example: query_length).",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help=f"Path to input parquet file (default: {DEFAULT_RAW_PATH}).",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help=f"Directory for output plots (default: {DEFAULT_PLOTS_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = args.raw_path
    plots_dir = args.plots_dir

    if not raw_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {raw_path}")

    if args.keyword is None or args.keyword == "all":
        selected = list(ANALYSES.keys())
    else:
        if args.keyword not in ANALYSES:
            valid = ", ".join(sorted(ANALYSES))
            raise ValueError(f"Unknown keyword {args.keyword!r}. Valid keywords: {valid}")
        selected = [args.keyword]

    for keyword in selected:
        ANALYSES[keyword](raw_path, plots_dir)


if __name__ == "__main__":
    main()
