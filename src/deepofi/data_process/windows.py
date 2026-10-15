from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .preprocessing import Normalizer

INPUT_FEATURES = {"of": 20, "lob": 40, "ofi": 10}


def list_dates(root, ticker):
    return sorted(path.name.split("_")[0] for path in (Path(root) / "features" / ticker).glob("*_ts.npy"))


def load_day(root, ticker, date, input_name="of", targets="short"):
    root = Path(root)
    x = np.load(root / "features" / ticker / f"{date}_{input_name}.npy")
    y = np.load(root / "targets" / ticker / f"{date}_{targets}.npy")
    mask = np.load(root / "targets" / ticker / f"{date}_mask_{targets}.npy")
    return x, y, mask


class WindowDataset(Dataset):
    """Samples (x[t-W+1 : t+1], y[t]) of shape [W, F] and [H] that never cross a day boundary.

    Features are normalized with `x_normalizer`; targets only if `y_normalizer` is given, so the
    test set can keep raw targets for R²_OS. Samples with any invalid horizon are dropped until
    the training loop supports masked loss.
    """

    def __init__(self, days, window=100, x_normalizer=None, y_normalizer=None):
        self.window = window
        self.features = []
        self.targets = []
        index = []

        for day, (x, y, mask) in enumerate(days):
            self.features.append(x_normalizer.transform(x) if x_normalizer else x.astype(np.float32))
            self.targets.append(y_normalizer.transform(y) if y_normalizer else y.astype(np.float32))
            valid = mask.all(axis=1)
            valid[: window - 1] = False
            positions = np.flatnonzero(valid)
            index.append(np.column_stack([np.full(len(positions), day), positions]))

        self.index = np.concatenate(index) if index else np.empty((0, 2), dtype=np.int64)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        day, t = self.index[i]
        x = self.features[day][t - self.window + 1 : t + 1]
        y = self.targets[day][t]
        return torch.from_numpy(x), torch.from_numpy(y)


def make_loaders(root, ticker, train_dates, val_dates, test_dates, input_name="of", window=100, batch_size=256, max_samples_per_epoch=None, seed=0):
    """Returns train/val/test loaders and the target normalizer for mapping predictions back to dollars."""
    train_days = [load_day(root, ticker, date, input_name) for date in train_dates]
    val_days = [load_day(root, ticker, date, input_name) for date in val_dates]
    test_days = [load_day(root, ticker, date, input_name) for date in test_dates]

    x_normalizer = Normalizer().fit([x for x, _, _ in train_days])
    y_normalizer = Normalizer().fit([y[mask.all(axis=1)] for _, y, mask in train_days])

    train = WindowDataset(train_days, window, x_normalizer, y_normalizer)
    val = WindowDataset(val_days, window, x_normalizer, y_normalizer)
    test = WindowDataset(test_days, window, x_normalizer)

    generator = torch.Generator().manual_seed(seed)
    num_samples = min(len(train), max_samples_per_epoch) if max_samples_per_epoch else len(train)
    sampler = RandomSampler(train, num_samples=num_samples, generator=generator)
    loaders = (
        DataLoader(train, batch_size=batch_size, sampler=sampler),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )
    return loaders, y_normalizer
