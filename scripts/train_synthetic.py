"""Train and evaluate one model on synthetic data (until the IEX dataset is ready).

    python scripts/train_synthetic.py --model lstm --epochs 5
"""
import argparse

import torch

from deepofi.data_process.synthetic import SYNTHETIC_ROOT, generate
from deepofi.data_process.windows import INPUT_FEATURES, list_dates, make_loaders
from deepofi.engine import train
from deepofi.evaluation.metrics import calculate_metrics, predict
from deepofi.models import MODELS, get_fit, get_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--input", choices=INPUT_FEATURES, default="of")
    parser.add_argument("--ticker", default="SYN1")
    parser.add_argument("--epochs", type=int, default=None, help="override TRAINING_PARAMS['epochs']")
    parser.add_argument("--learning-rate", type=float, default=None, help="override TRAINING_PARAMS['learning_rate']")
    parser.add_argument("--max-samples", type=int, default=50_000, help="random training windows per epoch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    dates = list_dates(SYNTHETIC_ROOT, args.ticker)
    if not dates:
        dates = generate()
    # Same order as the rolling scheme in the paper: validation, then training, then test.
    val_dates, train_dates, test_dates = dates[:1], dates[1:6], dates[6:]

    (train_loader, val_loader, test_loader), y_normalizer = make_loaders(
        SYNTHETIC_ROOT, args.ticker, train_dates, val_dates, test_dates,
        input_name=args.input, max_samples_per_epoch=args.max_samples, seed=args.seed,
    )

    model, params = get_model(args.model, n_features=INPUT_FEATURES[args.input])
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()):,} parameters")
    fit = get_fit(args.model)
    if fit:
        model, _ = fit(model, train_loader, val_loader, **params)
    else:
        model, _ = train(
            model, train_loader, val_loader,
            learning_rate=args.learning_rate or params["learning_rate"],
            weights_path=f"{args.model}_{args.input}_synthetic.pt",
            epochs=args.epochs or params.get("epochs", 50),
            patience=params.get("patience", 5),
            device=args.device,
            save_weights=False,
        )

    y_true, y_pred = predict(model, test_loader, device="cpu" if fit else args.device)
    y_pred = torch.from_numpy(y_normalizer.inverse(y_pred.double().numpy()))
    metrics = calculate_metrics(y_true, y_pred)
    print("horizon  R2_OS, %")
    for k, r2 in enumerate(metrics["r2_os"], start=1):
        print(f"{k:7d}  {100 * r2:7.2f}")


if __name__ == "__main__":
    main()
