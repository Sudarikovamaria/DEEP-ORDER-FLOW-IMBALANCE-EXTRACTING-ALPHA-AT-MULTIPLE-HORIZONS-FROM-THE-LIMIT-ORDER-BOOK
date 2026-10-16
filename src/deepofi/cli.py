import argparse
import csv
from pathlib import Path

import torch

from .config import load_config
from .data_process.dataset import make_window_loaders
from .data_process.splits import get_splits
from .data_process.targets import SHORT_HORIZONS
from .data_process.windows import INPUT_FEATURES
from .engine import load_weights, train
from .evaluation.metrics import evaluate, get_results_dir, predict, save_predictions
from .models import MODELS, get_fit, get_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("train", "evaluate", "predict"):
        command_parser = commands.add_parser(command)
        command_parser.add_argument("--model", choices=MODELS, required=True)
        command_parser.add_argument("--data-root", type=Path, required=True)
        command_parser.add_argument("--ticker", type=str.upper, required=True)
        command_parser.add_argument("--input", choices=INPUT_FEATURES, default="of")
        command_parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
        command_parser.add_argument("--save-forecasts", action="store_true")
        command_parser.add_argument("--sgd", action="store_true",
                                    help="train a closed-form model (ARX) with Adam like the paper; saved as {model}_sgd")
        if command == "train":
            command_parser.add_argument("--save-weights", action=argparse.BooleanOptionalAction, default=True)
            command_parser.add_argument("--epochs", type=int, default=None, help="cap on epochs; overrides the model and config value")
            command_parser.add_argument("--learning-rate", type=float, default=None, help="overrides the model's learning rate")

    return parser.parse_args()


def get_weights_path(args, window_name):
    return PROJECT_ROOT / "weights" / args.run_name / args.input / args.ticker / f"{window_name}.pt"


def check_files(args, windows):
    dates = sorted({date for window in windows for part in ("val_dates", "train_dates", "test_dates") for date in window[part]})

    for date in dates:
        for folder, extension in (("book", "parquet"), ("meta", "json")):
            path = args.data_root / folder / args.ticker / f"{date}.{extension}"
            if not path.is_file():
                raise FileNotFoundError(f"Missing prepared data: {path}")

    if args.command != "train":
        for window in windows:
            path = get_weights_path(args, window["name"])
            if not path.is_file():
                raise FileNotFoundError(f"Missing weights: {path}")


def save_training(model, history, args, window_name):
    if args.save_weights:
        path = get_weights_path(args, window_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        torch.save(state, path)
        print(f"Weights saved: {path}")

    history_path = get_results_dir(args.run_name, args.input, args.ticker, window_name) / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"Training history saved: {history_path}")


def run_window(args, window, config):
    settings = config["training"]
    torch.manual_seed(settings["seed"])
    model, params = get_model(args.model, n_features=INPUT_FEATURES[args.input], n_horizons=len(SHORT_HORIZONS))
    fit = None if args.sgd else get_fit(args.model)
    device = "cpu" if fit else args.device
    print(f"{args.run_name} | {args.input} | {args.ticker} | {window['name']}")

    loaders, y_normalizer, dt_ms = make_window_loaders(
        root=args.data_root, ticker=args.ticker,
        val_dates=window["val_dates"], train_dates=window["train_dates"], test_dates=window["test_dates"],
        input_name=args.input, config=config,
        window=settings["window"], batch_size=settings["batch_size"],
        max_samples_per_epoch=None if fit else settings.get("max_samples_per_epoch"),
        max_val_samples=None if fit else settings.get("max_val_samples"), seed=settings["seed"],
    )
    train_loader, val_loader, test_loader = loaders
    print(f"Delta t: {dt_ms:.3f} ms")

    if args.command == "train":
        if fit:
            model, history = fit(model, train_loader, val_loader, **params)
        else:
            model, history = train(
                model, train_loader, val_loader, learning_rate=args.learning_rate or params["learning_rate"],
                weights_path=get_weights_path(args, window["name"]),
                epochs=args.epochs or params.get("epochs", settings["epochs"]), patience=params.get("patience", settings["patience"]),
                device=device, save_weights=False,
            )
        save_training(model, history, args, window["name"])
    else:
        model = load_weights(model, weights_path=get_weights_path(args, window["name"]), device=device)

    if args.command == "predict":
        y_true, y_pred = predict(model, test_loader, device=device, y_normalizer=y_normalizer)
        save_predictions(args.run_name, y_true, y_pred, SHORT_HORIZONS, args.input, args.ticker, window["name"])
        return

    metrics = evaluate(
        model=model, test_loader=test_loader, y_normalizer=y_normalizer, horizons=SHORT_HORIZONS,
        model_name=args.run_name, input_name=args.input, ticker=args.ticker, window_name=window["name"],
        device=device, save_forecasts=args.save_forecasts, dates=window["test_dates"],
    )
    mean_r2 = sum(metrics["r2_os"]) / len(metrics["r2_os"])
    print(f"Mean R2_OS across horizons: {100 * mean_r2:.3f}%")


def main():
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.run_name = f"{args.model}_sgd" if args.sgd else args.model
    config = load_config()
    windows = get_splits()
    check_files(args, windows)

    for window in windows:
        run_window(args, window, config)


if __name__ == "__main__":
    main()