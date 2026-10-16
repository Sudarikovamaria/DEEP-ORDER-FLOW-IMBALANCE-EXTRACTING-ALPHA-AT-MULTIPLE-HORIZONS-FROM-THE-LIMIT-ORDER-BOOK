import csv
from pathlib import Path

import torch

from ..engine import get_device


def calculate_metrics(y_true, y_pred, benchmark=None):
    """MSE and R2_OS per horizon; the benchmark defaults to the mean return of y_true."""
    y_true = y_true.detach().cpu().double()
    y_pred = y_pred.detach().cpu().double()
    benchmark = y_true.mean(dim=0) if benchmark is None else benchmark
    mse = ((y_true - y_pred) ** 2).mean(dim=0)
    benchmark_mse = ((y_true - benchmark) ** 2).mean(dim=0)
    r2_os = torch.where(benchmark_mse > 0, 1 - mse / benchmark_mse, float("nan"))
    return {"mse": mse.tolist(), "r2_os": r2_os.tolist(), "n_obs": len(y_true)}


def daily_metrics(y_true, y_pred, days, dates):
    """R2_OS for each test day against the mean return of the whole test week (Tables 3-4 need daily values)."""
    week_mean = y_true.detach().cpu().double().mean(dim=0)
    days = torch.as_tensor(days)
    return {
        date: calculate_metrics(y_true[days == day], y_pred[days == day], benchmark=week_mean)
        for day, date in enumerate(dates)
        if (days == day).any()
    }


def predict(model, test_loader, device="auto", y_normalizer=None):
    device = get_device(device)
    model = model.to(device)
    model.eval()
    targets = []
    predictions = []

    with torch.no_grad():
        for X, y in test_loader:
            predictions.append(model(X.to(device)).detach().cpu())
            targets.append(y.detach().cpu())

    y_true = torch.cat(targets, dim=0)
    y_pred = torch.cat(predictions, dim=0)

    if y_normalizer is not None:
        y_pred = torch.from_numpy(y_normalizer.inverse(y_pred.double().numpy()))

    return y_true, y_pred


def get_results_dir(model_name, input_name, ticker, window_name):
    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / "results" / model_name / input_name / ticker / window_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_metrics(model_name, metrics, horizons, input_name, ticker, window_name, daily=None):
    """Rows with date "week" cover the whole test week; one row per test day and horizon follows."""
    metrics_path = get_results_dir(model_name, input_name, ticker, window_name) / "metrics.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["date", "horizon", "mse", "r2_os", "n_obs"])

        for date, values in {"week": metrics, **(daily or {})}.items():
            for horizon, mse, r2_os in zip(horizons, values["mse"], values["r2_os"], strict=True):
                writer.writerow([date, horizon, mse, r2_os, values["n_obs"]])

    print(f"Metrics saved: {metrics_path}")
    return metrics_path


def save_predictions(model_name, y_true, y_pred, horizons, input_name, ticker, window_name):
    predictions_path = get_results_dir(model_name, input_name, ticker, window_name) / "predictions.csv"

    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample", "horizon", "y_true", "y_pred"])

        for sample, (targets, predictions) in enumerate(zip(y_true, y_pred, strict=True), start=1):
            for horizon, target, prediction in zip(horizons, targets.tolist(), predictions.tolist(), strict=True):
                writer.writerow([sample, horizon, target, prediction])

    print(f"Predictions saved: {predictions_path}")
    return predictions_path


def evaluate(model, test_loader, y_normalizer, horizons, model_name, input_name, ticker, window_name, device="auto", save_forecasts=False, dates=None):
    """`dates` are the test days in loader order; with them metrics.csv also gets daily R2_OS."""
    y_true, y_pred = predict(model, test_loader, device=device, y_normalizer=y_normalizer)
    metrics = calculate_metrics(y_true, y_pred)
    daily = daily_metrics(y_true, y_pred, test_loader.dataset.index[:, 0], dates) if dates else None
    save_metrics(model_name, metrics, horizons, input_name, ticker, window_name, daily)

    if save_forecasts:
        save_predictions(model_name, y_true, y_pred, horizons, input_name, ticker, window_name)

    return metrics