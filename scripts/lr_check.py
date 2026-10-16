"""Learning-rate check for the lr-1e-5 models: LSTM on BAC window 1, 4 epochs at 1e-5 vs 1e-4.

Same seed and loaders; the rate with the lower validation MSE is written to data/lr_choice.txt
(chosen on validation only, never on test). Test R2_OS is printed for information.

    python scripts/lr_check.py
"""
import time
import numpy as np
import torch
from deepofi.config import load_config
from deepofi.data_process.dataset import make_window_loaders
from deepofi.data_process.splits import get_splits
from deepofi.engine import train
from deepofi.evaluation.metrics import calculate_metrics, daily_metrics, predict
from deepofi.models import get_model

cfg = load_config(); s = cfg["training"]; w = get_splits()[0]
(tr, va, te), yn, dt = make_window_loaders("data", "BAC", w["val_dates"], w["train_dates"], w["test_dates"], "of", config=cfg,
                                           window=100, batch_size=256, max_samples_per_epoch=s["max_samples_per_epoch"],
                                           max_val_samples=s["max_val_samples"], seed=0)
print(f"loaders ready, delta t {dt:.0f} ms", flush=True)
best = {}
for lr in (1e-5, 1e-4):
    torch.manual_seed(0)
    model, _ = get_model("lstm", n_features=20)
    t0 = time.time()
    model, hist = train(model, tr, va, learning_rate=lr, weights_path="unused.pt", epochs=4, patience=100, device="auto", save_weights=False)
    y_true, y_pred = predict(model, te, device="auto", y_normalizer=yn)
    week = calculate_metrics(y_true, y_pred)["r2_os"]
    daily = daily_metrics(y_true, y_pred, te.dataset.index[:, 0], w["test_dates"])
    daily_mean = np.mean([np.mean(v["r2_os"]) for v in daily.values()])
    val = [round(h["val_mse"], 5) for h in hist]
    best[lr] = min(val)
    print(f"RESULT lr={lr:g}: val_mse by epoch {val} | change {100 * (val[-1] / val[0] - 1):+.2f}% after epoch 1 | "
          f"test R2_OS week {100 * np.mean(week):+.3f}% daily {100 * daily_mean:+.3f}% | {time.time() - t0:.0f} s", flush=True)
choice = min(best, key=best.get)  # chosen on validation only
open("data/lr_choice.txt", "w").write(f"{choice:g}\n")
print(f"CHOICE lr={choice:g} (best val_mse {best})", flush=True)
