import csv
from pathlib import Path

import torch

from ..engine import get_device


def calculate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().double()
    y_pred = y_pred.detach().cpu().double()
    mse = ((y_true - y_pred) ** 2).mean(dim=0)
    benchmark_mse = ((y_true - y_true.mean(dim=0)) ** 2).mean(dim=0)
    r2_os = torch.where(benchmark_mse > 0, 1 - mse / benchmark_mse, float("nan"))
    return {"mse": mse.tolist(), "r2_os": r2_os.tolist()}


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


def save_metrics(model_name, metrics, horizons, input_name, ticker, window_name):
    metrics_path = get_results_dir(model_name, input_name, ticker, window_name) / "metrics.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["horizon", "mse", "r2_os"])

        for horizon, mse, r2_os in zip(horizons, metrics["mse"], metrics["r2_os"], strict=True):
            writer.writerow([horizon, mse, r2_os])

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


def evaluate(model, test_loader, y_normalizer, horizons, model_name, input_name, ticker, window_name, device="auto", save_forecasts=False):
    y_true, y_pred = predict(model, test_loader, device=device, y_normalizer=y_normalizer)
    metrics = calculate_metrics(y_true, y_pred)
    save_metrics(model_name, metrics, horizons, input_name, ticker, window_name)

    if save_forecasts:
        save_predictions(model_name, y_true, y_pred, horizons, input_name, ticker, window_name)

    return metrics