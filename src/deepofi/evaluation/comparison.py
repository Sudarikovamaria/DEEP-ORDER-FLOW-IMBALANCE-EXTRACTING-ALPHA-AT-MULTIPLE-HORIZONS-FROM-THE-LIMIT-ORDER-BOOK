import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..data_process.splits import get_splits
from ..data_process.windows import INPUT_FEATURES
from ..models import MODELS

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_metrics(model_names, input_name, ticker=None):
    tables = []
    expected_windows = {window["name"] for window in get_splits()}
    reference_pairs = None
    reference_horizons = None

    for model_name in model_names:
        base = PROJECT_ROOT / "results" / model_name / input_name
        paths = sorted(base.glob(f"{ticker or '*'}/window_*/metrics.csv"))
        paths = [path for path in paths if path.parent.name in expected_windows]
        if not paths:
            raise FileNotFoundError(f"No metrics for {model_name}, input {input_name}, ticker {ticker or 'all'}")

        pairs = set()
        for path in paths:
            ticker_name = path.parent.parent.name
            window_name = path.parent.name
            table = pd.read_csv(path).sort_values("horizon")
            horizons = table["horizon"].tolist()

            if reference_horizons is None:
                reference_horizons = horizons
            elif horizons != reference_horizons:
                raise ValueError(f"Forecast horizons differ: {path}")

            table = table[["horizon", "mse", "r2_os"]].copy()
            table["model"] = model_name
            table["ticker"] = ticker_name
            table["window"] = window_name
            tables.append(table)
            pairs.add((ticker_name, window_name))

        tickers = {name for name, _ in pairs}
        if pairs != {(name, window) for name in tickers for window in expected_windows}:
            raise ValueError(f"Metrics for all three windows are required: {model_name}")

        if reference_pairs is None:
            reference_pairs = pairs
        elif pairs != reference_pairs:
            raise ValueError("Models must have metrics for the same tickers and windows")

    return pd.concat(tables, ignore_index=True)


def compare_models(model_names, input_name="of", ticker=None):
    model_names = list(dict.fromkeys(model_names))
    metrics = load_metrics(model_names, input_name, ticker)
    comparison = metrics.groupby(["model", "horizon"], as_index=False)[["mse", "r2_os"]].mean()

    output_dir = PROJECT_ROOT / "results" / "model_comparison" / input_name / (ticker or "all_tickers")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "comparison.csv", index=False)

    for metric, label in (("mse", "Mean MSE"), ("r2_os", "Mean R²_OS")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_name in model_names:
            table = comparison[comparison["model"] == model_name].sort_values("horizon")
            ax.plot(table["horizon"], table[metric], marker="o", label=model_name)

        if metric == "r2_os":
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Horizon (fraction of average price-change time)")
        ax.set_ylabel(label)
        ax.set_title(f"Model comparison: {input_name.upper()}")
        ax.set_xticks(sorted(comparison["horizon"].unique()))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}_comparison.png", dpi=300)
        plt.close(fig)

    print(f"Comparison saved: {output_dir}")
    return comparison


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--input", choices=INPUT_FEATURES, default="of")
    parser.add_argument("--ticker", type=str.upper, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_models(args.models, args.input, args.ticker)