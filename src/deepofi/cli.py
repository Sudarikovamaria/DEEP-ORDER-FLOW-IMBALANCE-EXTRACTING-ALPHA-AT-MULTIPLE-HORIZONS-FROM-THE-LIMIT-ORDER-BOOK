import argparse

from .engine import load_weights, train
from .evaluation.metrics import evaluate, predict, save_predictions
from .models import MODELS, get_model


def parse_args():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("train", "evaluate", "predict"):
        command_parser = commands.add_parser(command)
        command_parser.add_argument("--model", choices=MODELS, required=True)
        command_parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
        if command == "train":
            command_parser.add_argument("--save-weights", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def run_training(args, train_loader, val_loader):
    model, params = get_model(args.model)
    model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=params["learning_rate"],
        weights_path=f"{args.model}.pt",
        epochs=params.get("epochs", 50),
        patience=params.get("patience", 5),
        device=args.device,
        save_weights=args.save_weights,
    )
    return model, history


def run_evaluation(args, test_loader, horizons):
    return evaluate(args.model, test_loader, horizons, device=args.device)


def run_prediction(args, test_loader, horizons):
    model, _ = get_model(args.model)
    model = load_weights(model, weights_path=f"{args.model}.pt", device=args.device)
    y_true, y_pred = predict(model, test_loader, device=args.device)
    save_predictions(args.model, y_true, y_pred, horizons)
    return y_pred


if __name__ == "__main__":
    args = parse_args()
    print(f"Selected command: {args.command} | Model: {args.model}")