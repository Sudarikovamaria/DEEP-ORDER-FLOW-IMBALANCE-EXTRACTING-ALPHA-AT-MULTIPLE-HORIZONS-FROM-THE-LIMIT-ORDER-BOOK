"""Figures and tables for the presentation, built from results/ (paper analogues, HANDOFF §8).

    python scripts/make_report.py                     # writes results/report/

Uses the daily rows of metrics.csv. A model enters a figure only for tickers where all three windows
are finished, so the script can be rerun as teammates push results. Outputs:
  fig2_term_structure.png     mean R2_OS by horizon per model (Figure 2): daily mean per stock, then stocks
  fig3_ranks.png / .csv       average rank of each model by horizon (Figure 3)
  table4_significance.csv     significant positive minus negative stocks at 1%, HC0 t-stats (Table 4)
  fig4_lstm_vs_cnn_lstm.png   per-stock R2_OS of LSTM vs CNN-LSTM (Figure 4)
  fig5_cross_section.png      per-stock R2_OS vs tick size and log(updates / price changes) (Figures 5, 7)
  table5_cross_section.csv    univariate OLS of per-stock R2_OS on characteristics, HC0 t-stats (Table 5)
  figB_arx_of_vs_ofi.png      ARX on OF vs OFI (Appendix B.1)
  table_data.csv              tickers, activity, delta t and tick size over the period
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy import stats

from deepofi.data_process.splits import get_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = {"arx": "ARX", "mlp": "MLP", "lstm": "LSTM", "lstm3": "LSTM (3)", "lstm_mlp": "LSTM-MLP", "cnn_lstm": "CNN-LSTM"}
COLORS = {"arx": "#7a7f87", "mlp": "#c07a2c", "lstm": "#1f77b4", "lstm3": "#5aa6d6", "lstm_mlp": "#2a9d6f", "cnn_lstm": "#b8336a"}
WINDOWS = {w["name"] for w in get_splits()}
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data", help="dataset with meta/ for stock characteristics")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "report")
    return parser.parse_args()


def load_daily(results, model, input_name):
    """Daily rows for tickers with all three windows: ticker, window, date, horizon, r2_os, mse, n_obs."""
    frames = []
    for ticker_dir in sorted((results / model / input_name).glob("*")):
        paths = [ticker_dir / w / "metrics.csv" for w in sorted(WINDOWS)]
        if not all(p.exists() for p in paths):
            continue
        for path in paths:
            table = pd.read_csv(path)
            if "date" not in table:
                continue
            table = table[table["date"] != "week"].copy()
            table["ticker"], table["window"] = ticker_dir.name, path.parent.name
            frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def hc0_t(values):
    """t-statistic of the mean with White (HC0) standard errors, as a regression on a constant."""
    x = np.asarray(values, float)
    e = x - x.mean()
    return x.mean() * len(x) / np.sqrt((e ** 2).sum())


def term_structure(daily):
    return daily.groupby(["ticker", "horizon"])["r2_os"].mean().groupby("horizon").mean() * 100


def per_stock(daily):
    """One R2_OS per stock: average over the term structure per day, then over days (Section 4.2)."""
    return daily.groupby(["ticker", "date"])["r2_os"].mean().groupby("ticker").mean() * 100


def significance_table(daily_by_model):
    rows = []
    for model, daily in daily_by_model.items():
        n_days = daily.groupby(["ticker", "horizon"]).size().min()
        crit = stats.t.ppf(0.995, n_days - 1)
        t = daily.groupby(["ticker", "horizon"])["r2_os"].apply(hc0_t).unstack()
        row = {"model": MODELS[model], "stocks": t.shape[0], "test_days": int(n_days)}
        for h in t.columns:
            row[f"{h:.1f}"] = int((t[h] > crit).sum() - (t[h] < -crit).sum())
        avg = daily.groupby(["ticker", "date"])["r2_os"].mean().groupby("ticker").apply(hc0_t)
        row["Avg"] = int((avg > crit).sum() - (avg < -crit).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def rank_table(daily_by_model):
    """Figure 3: rank models per ticker, day and horizon (1 = best), average across tickers per day, then days."""
    stacked = pd.concat([d.assign(model=m) for m, d in daily_by_model.items()])
    keys = ["ticker", "date", "horizon"]
    complete = stacked.groupby(keys)["model"].transform("nunique") == len(daily_by_model)
    stacked = stacked[complete]
    stacked["rank"] = stacked.groupby(keys)["r2_os"].rank(ascending=False)
    daily_rank = stacked.groupby(["model", "date", "horizon"])["rank"].mean()
    ranks = daily_rank.groupby(["model", "horizon"]).mean().unstack()
    ranks["Avg"] = daily_rank.groupby(["model", "date"]).mean().groupby("model").mean()
    ranks.index = [MODELS[m] for m in ranks.index]
    return ranks.sort_values("Avg")


def stock_characteristics(data_root, tickers, period):
    rows = []
    for ticker in tickers:
        metas = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(data_root / "meta" / ticker / "2026*.json")))]
        metas = [m for m in metas if period[0] <= m["date"] <= period[1] and m.get("price_changes")]
        frame = pd.DataFrame(metas)
        rows.append({
            "ticker": ticker,
            "tick_size": frame["tick_size_frac"].median(),
            "updates_per_day": frame["rows_kept"].median(),
            "trades_per_day": frame["trades"].median(),
            "price_changes_per_day": frame["price_changes"].median(),
            "delta_t_ms": 2.34e7 / frame["price_changes"].median(),
            "median_spread_ticks": frame["median_spread_ticks"].median(),
        })
    table = pd.DataFrame(rows).set_index("ticker")
    table["log_updates"] = np.log(table["updates_per_day"])
    table["log_trades"] = np.log(table["trades_per_day"])
    table["log_price_changes"] = np.log(table["price_changes_per_day"])
    table["log_updates_per_price_change"] = np.log(table["updates_per_day"] / table["price_changes_per_day"])
    return table


def cross_section_table(r2_by_model, characteristics):
    rows = []
    regressors = ["tick_size", "log_updates", "log_trades", "log_price_changes", "log_updates_per_price_change"]
    for model, r2 in r2_by_model.items():
        frame = characteristics.join(r2.rename("r2"), how="inner")
        for name in regressors:
            fit = sm.OLS(frame["r2"], sm.add_constant(frame[name])).fit(cov_type="HC0")
            rows.append({"model": MODELS[model], "stocks": len(frame), "regressor": name, "coef": fit.params[name],
                         "t_stat": fit.tvalues[name], "adj_r2": fit.rsquared_adj,
                         "spearman": stats.spearmanr(frame[name], frame["r2"]).correlation})
    return pd.DataFrame(rows)


def save_term_structure(daily_by_model, path, title):
    curves = {model: term_structure(daily) for model, daily in daily_by_model.items()}
    counts = {model: daily["ticker"].nunique() for model, daily in daily_by_model.items()}
    same_count = len(set(counts.values())) == 1
    fig, ax = plt.subplots(figsize=(11, 5.6))
    # Legend order follows the curves at the longest horizon, top to bottom.
    for model in sorted(curves, key=lambda m: curves[m].iloc[-1], reverse=True):
        curve = curves[model]
        label = MODELS[model] if same_count else f"{MODELS[model]} ({counts[model]} stocks)"
        ax.plot(curve.index, curve.values, marker="o", color=COLORS[model], label=label, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(curve.index)
    ax.set_xlabel("Horizon (fraction of an average price change, Δt)")
    ax.set_ylabel("Mean R²$_{OS}$ (%)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    universe = yaml.safe_load((PROJECT_ROOT / "configs" / "tickers.yaml").read_text())
    all_tickers = [t for group in universe["tickers"].values() for t in group]

    of = {m: d for m in MODELS if not (d := load_daily(args.results, m, "of")).empty}
    print("OF results:", {MODELS[m]: d["ticker"].nunique() for m, d in of.items()})
    if not of:
        raise SystemExit("No complete results found")

    # Neural models run on 6 tickers; compare every model on the tickers they all share.
    common = set.intersection(*(set(d["ticker"]) for d in of.values()))
    of_common = {m: d[d["ticker"].isin(common)] for m, d in of.items()}
    print(f"Common tickers ({len(common)}): {', '.join(sorted(common))}")

    save_term_structure(of_common, args.out / "fig2_term_structure.png", f"Out-of-sample R²$_{{OS}}$ by horizon, OF input ({len(common)} common stocks)")
    if "arx" in of:
        save_term_structure({"arx": of["arx"]}, args.out / "fig2_arx_all_stocks.png", "ARX on OF, all 11 stocks")

    significance = significance_table(of)
    significance.to_csv(args.out / "table4_significance.csv", index=False)
    print("\nTable 4 analogue (significant positive minus negative stocks, 1%):\n" + significance.to_string(index=False))

    if len(of_common) > 1:
        ranks = rank_table(of_common)
        ranks.round(2).to_csv(args.out / "fig3_ranks.csv")
        fig, ax = plt.subplots(figsize=(10, 0.6 * len(ranks) + 1.8))
        image = ax.imshow(ranks.values, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(ranks.shape[1]), [c if c == "Avg" else f"{c:.1f}" for c in ranks.columns])
        ax.set_yticks(range(len(ranks)), ranks.index)
        for (i, j), value in np.ndenumerate(ranks.values):
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="white", fontsize=10)
        ax.set_xlabel("Horizon (Δt)")
        ax.set_title(f"Average rank by horizon (1 = best), {len(common)} stocks")
        fig.colorbar(image, ax=ax, fraction=0.025)
        fig.tight_layout()
        fig.savefig(args.out / "fig3_ranks.png", dpi=200)
        plt.close(fig)

    r2 = {m: per_stock(d) for m, d in of.items()}
    if "lstm" in r2 and "cnn_lstm" in r2:
        both = pd.concat([r2["lstm"].rename("lstm"), r2["cnn_lstm"].rename("cnn_lstm")], axis=1).dropna()
        slope, intercept = np.polyfit(both["lstm"], both["cnn_lstm"], 1)
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(both["lstm"], both["cnn_lstm"], s=60, color=COLORS["cnn_lstm"])
        for ticker, row in both.iterrows():
            ax.annotate(ticker, (row["lstm"], row["cnn_lstm"]), xytext=(4, 4), textcoords="offset points", fontsize=10)
        lims = [min(both.min()), max(both.max())]
        ax.plot(lims, lims, color="black", linewidth=0.8, linestyle="--", label="y = x")
        ax.plot(lims, np.polyval([slope, intercept], lims), color=COLORS["cnn_lstm"], label=f"fit: y = {slope:.2f}x {intercept:+.2f}")
        ax.set_xlabel("R²$_{OS}$ LSTM (%)")
        ax.set_ylabel("R²$_{OS}$ CNN-LSTM (%)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.out / "fig4_lstm_vs_cnn_lstm.png", dpi=200)
        plt.close(fig)

    characteristics = stock_characteristics(args.data_root, all_tickers, universe["period"])
    groups = {t: g.replace("_tick", "") for g, ts in universe["tickers"].items() for t in ts}
    data_table = characteristics.assign(group=pd.Series(groups)).round(3)
    data_table.to_csv(args.out / "table_data.csv")

    cross = cross_section_table({m: r for m, r in r2.items() if m in ("arx", "lstm", "cnn_lstm")}, characteristics)
    cross.round(4).to_csv(args.out / "table5_cross_section.csv", index=False)
    print("\nTable 5 analogue (univariate, HC0):\n" + cross.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    all_r2 = pd.DataFrame(r2)
    for ax, name, label in ((axes[0], "tick_size", "Tick size (fraction of time with a one-tick spread)"),
                            (axes[1], "log_updates_per_price_change", "log(updates / price changes)")):
        for model in r2:
            frame = characteristics.join(r2[model].rename("r2"), how="inner")
            ax.scatter(frame[name], frame["r2"], s=45, color=COLORS[model], label=f"{MODELS[model]} ({len(frame)})", alpha=0.85)
        for ticker, top in all_r2.max(axis=1).items():
            ax.annotate(ticker, (characteristics.loc[ticker, name], top), xytext=(0, 7), textcoords="offset points",
                        fontsize=9, color="#555", ha="center")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel(label)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Per-stock mean R²$_{OS}$ (%)")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Predictability and stock characteristics, OF input")
    fig.tight_layout()
    fig.savefig(args.out / "fig5_cross_section.png", dpi=200)
    plt.close(fig)

    ofi = load_daily(args.results, "arx", "ofi")
    if "arx" in of and not ofi.empty:
        fig, ax = plt.subplots(figsize=(10, 5.6))
        for daily, label, style in ((of["arx"], "ARX, OF (20 features × 100 lags)", "-"), (ofi, "ARX, OFI (10 features × 100 lags)", "--")):
            curve = term_structure(daily)
            ax.plot(curve.index, curve.values, marker="o", linestyle=style, color=COLORS["arx"], label=label, linewidth=2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(curve.index)
        ax.set_xlabel("Horizon (Δt)")
        ax.set_ylabel("Mean R²$_{OS}$ (%)")
        ax.set_title(f"ARX: order flow vs order flow imbalance ({ofi['ticker'].nunique()} stocks)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.out / "figB_arx_of_vs_ofi.png", dpi=200)
        plt.close(fig)

    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
