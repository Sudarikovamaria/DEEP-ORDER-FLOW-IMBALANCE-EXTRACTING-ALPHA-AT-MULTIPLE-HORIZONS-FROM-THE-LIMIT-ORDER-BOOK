from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..models import MODELS

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_metrics(model_names):
    tables = []

    for model_name in model_names:
        metrics_path = PROJECT_ROOT / "results" / model_name / "metrics.csv"
        table = pd.read_csv(metrics_path).sort_values("horizon")
        if tables and table["horizon"].tolist() != tables[0]["horizon"].tolist():
            raise ValueError("Models must have the same forecast horizons")
        table.insert(0, "model", model_name)
        tables.append(table)

    return pd.concat(tables, ignore_index=True)


def compare_models(model_names):
    model_names = list(model_names)
    comparison = load_metrics(model_names)
    output_dir = PROJECT_ROOT / "results" / "model_comparison"

    for metric, label in (("mse", "MSE"), ("r2_os", "R²_OS")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_name in model_names:
            table = comparison[comparison["model"] == model_name]
            ax.plot(table["horizon"], table[metric], marker="o", label=model_name)

        if metric == "r2_os":
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Horizon (fraction of average price-change time)")
        ax.set_ylabel(label)
        ax.set_title(f"Model comparison: {label}")
        ax.set_xticks(comparison["horizon"].unique())
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}_comparison.png", dpi=300)
        plt.close(fig)

    print(f"Comparison figures saved: {output_dir}")
    return comparison


if __name__ == "__main__":
    compare_models(MODELS)