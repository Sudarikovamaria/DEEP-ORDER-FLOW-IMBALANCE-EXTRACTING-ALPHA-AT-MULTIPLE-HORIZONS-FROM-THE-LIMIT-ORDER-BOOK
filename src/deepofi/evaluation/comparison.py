import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..data_process.splits import get_splits
from ..data_process.windows import INPUT_FEATURES
from ..models import MODELS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_TICKERS = ("BAC", "VZ", "NKE", "USB", "OXY", "UBER")


def load_metrics(model_names, input_name, ticker=None, tickers=None):
    """Weekly metrics for the six experiment tickers, or an explicitly selected set.

    Every selected ticker must have all three windows for every model; extra tickers are ignored.
    """
    expected_windows = {window["name"] for window in get_splits()}
    wanted = {ticker} if ticker else set(tickers or EXPERIMENT_TICKERS)
    reference_horizons = None
    per_model = {}

    for model_name in model_names:
        base = PROJECT_ROOT / "results" / model_name / input_name
        paths = [
            path for path in sorted(base.glob("*/window_*/metrics.csv"))
            if path.parent.name in expected_windows and path.parent.parent.name in wanted
        ]
        tables = []
        for path in paths:
            table = pd.read_csv(path)
            if "date" in table:
                table = table[table["date"] == "week"]
            table = table.sort_values("horizon")
            horizons = table["horizon"].tolist()
            if reference_horizons is None:
                reference_horizons = horizons
            elif horizons != reference_horizons:
                raise ValueError(f"Forecast horizons differ: {path}")

            table = table[["horizon", "mse", "r2_os"]].copy()
            table["model"] = model_name
            table["ticker"] = path.parent.parent.name
            table["window"] = path.parent.name
            tables.append(table)
        if not tables:
            raise FileNotFoundError(f"No metrics for {model_name}, input {input_name}")
        per_model[model_name] = pd.concat(tables, ignore_index=True)

    complete = [
        {name for name, windows in metrics.groupby("ticker")["window"] if set(windows) == expected_windows}
        for metrics in per_model.values()
    ]
    common = set.intersection(*complete) & wanted
    missing = wanted - common
    if missing:
        raise ValueError(f"Missing complete three-window metrics for tickers: {sorted(missing)}")
    print(f"Comparing on {len(common)} tickers: {', '.join(sorted(common))}")

    metrics = pd.concat(per_model.values(), ignore_index=True)
    return metrics[metrics["ticker"].isin(common)]


def compare_models(model_names, input_name="of", ticker=None, tickers=None):
    model_names = list(dict.fromkeys(model_names))
    metrics = load_metrics(model_names, input_name, ticker, tickers)
    comparison = metrics.groupby(["model", "horizon"], as_index=False)[["mse", "r2_os"]].mean()

    output_dir = PROJECT_ROOT / "results" / "model_comparison" / input_name / (ticker or ("selected_tickers" if tickers else "all_tickers"))
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    (output_dir / "tickers.txt").write_text("\n".join(sorted(metrics["ticker"].unique())) + "\n")

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
    parser.add_argument("--tickers", nargs="+", type=str.upper, default=None, help="compare only on these tickers")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_models(args.models, args.input, args.ticker, args.tickers)