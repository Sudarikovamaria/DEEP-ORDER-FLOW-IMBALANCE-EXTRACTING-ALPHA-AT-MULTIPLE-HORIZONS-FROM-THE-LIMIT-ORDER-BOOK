import csv
from pathlib import Path

import torch

from ..engine import get_device, load_weights
from ..models import get_model


def calculate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().double()
    y_pred = y_pred.detach().cpu().double()
    mse = ((y_true - y_pred) ** 2).mean(dim=0)
    benchmark = y_true.mean(dim=0)
    benchmark_mse = ((y_true - benchmark) ** 2).mean(dim=0)
    r2_os = torch.where(benchmark_mse > 0, 1 - mse / benchmark_mse, float("nan"))
    return {"mse": mse.tolist(), "r2_os": r2_os.tolist()}


def save_metrics(model_name, metrics, horizons):
    project_root = Path(__file__).resolve().parents[3]
    metrics_path = project_root / "results" / model_name / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["horizon", "mse", "r2_os"])
        for horizon, mse, r2_os in zip(
            horizons, metrics["mse"], metrics["r2_os"], strict=True
        ):
            writer.writerow([horizon, mse, r2_os])

    print(f"Metrics saved: {metrics_path}")
    return metrics_path


def predict(model, test_loader, device="auto"):
    device = get_device(device)
    model = model.to(device)
    model.eval()
    targets = []
    predictions = []

    with torch.no_grad():
        for X, y in test_loader:
            batch_predictions = model(X.to(device))
            targets.append(y.detach().cpu())
            predictions.append(batch_predictions.cpu())

    return torch.cat(targets, dim=0), torch.cat(predictions, dim=0)


def save_predictions(model_name, y_true, y_pred, horizons):
    project_root = Path(__file__).resolve().parents[3]
    predictions_path = project_root / "results" / model_name / "predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample", "horizon", "y_true", "y_pred"])
        for sample, (targets, predictions) in enumerate(zip(y_true.tolist(), y_pred.tolist(), strict=True), start=1):
            for horizon, target, prediction in zip(horizons, targets, predictions, strict=True):
                writer.writerow([sample, horizon, target, prediction])

    print(f"Predictions saved: {predictions_path}")
    return predictions_path


def evaluate(model_name, test_loader, horizons, device="auto"):
    model, _ = get_model(model_name)
    model = load_weights(model, weights_path=f"{model_name}.pt", device=device)
    y_true, y_pred = predict(model, test_loader, device=device)
    save_predictions(model_name, y_true, y_pred, horizons)
    metrics = calculate_metrics(y_true, y_pred)
    save_metrics(model_name, metrics, horizons)
    return metrics
