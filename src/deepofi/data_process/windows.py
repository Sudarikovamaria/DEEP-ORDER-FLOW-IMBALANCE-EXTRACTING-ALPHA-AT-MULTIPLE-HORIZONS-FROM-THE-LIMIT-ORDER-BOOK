from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

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

    def __getitems__(self, indices):
        """A whole batch at once (DataLoader calls this instead of __getitem__ per sample).

        Windows are cut with one strided view per day; the result equals stacking __getitem__.
        """
        rows = self.index[np.asarray(indices)]
        x = np.empty((len(rows), self.window, self.features[0].shape[1]), dtype=np.float32)
        y = np.empty((len(rows), self.targets[0].shape[1]), dtype=np.float32)
        for day in np.unique(rows[:, 0]):
            take = np.flatnonzero(rows[:, 0] == day)
            t = rows[take, 1]
            windows = np.lib.stride_tricks.sliding_window_view(self.features[day], self.window, axis=0)  # [T - W + 1, F, W]
            x[take] = windows[t - self.window + 1].transpose(0, 2, 1)
            y[take] = self.targets[day][t]
        return torch.from_numpy(x), torch.from_numpy(y)


def _batch_as_is(batch):
    """Collate function for datasets whose __getitems__ already returns stacked tensors."""
    return batch


def make_loaders(root, ticker, train_dates, val_dates, test_dates, input_name="of", **kwargs):
    """Returns train/val/test loaders and the target normalizer for mapping predictions back to dollars."""
    return make_loaders_from_days(
        [load_day(root, ticker, date, input_name) for date in train_dates],
        [load_day(root, ticker, date, input_name) for date in val_dates],
        [load_day(root, ticker, date, input_name) for date in test_dates],
        **kwargs,
    )


def make_loaders_from_days(train_days, val_days, test_days, window=100, batch_size=256, max_samples_per_epoch=None, max_val_samples=None, seed=0):
    """Same as make_loaders for (x, y, mask) segments already in memory; windows never cross segments.

    max_val_samples keeps a fixed random subset of validation windows (same on every epoch) to make
    early stopping cheaper; the test set is always complete.
    """
    x_normalizer = Normalizer().fit([x for x, _, _ in train_days])
    y_normalizer = Normalizer().fit([y[mask.all(axis=1)] for _, y, mask in train_days])

    train = WindowDataset(train_days, window, x_normalizer, y_normalizer)
    val = WindowDataset(val_days, window, x_normalizer, y_normalizer)
    test = WindowDataset(test_days, window, x_normalizer)

    if max_val_samples and len(val) > max_val_samples:
        chosen = torch.randperm(len(val), generator=torch.Generator().manual_seed(seed + 1))[:max_val_samples]
        val = Subset(val, sorted(chosen.tolist()))

    generator = torch.Generator().manual_seed(seed)
    num_samples = min(len(train), max_samples_per_epoch) if max_samples_per_epoch else len(train)
    sampler = RandomSampler(train, num_samples=num_samples, generator=generator)
    loaders = (
        DataLoader(train, batch_size=batch_size, sampler=sampler, collate_fn=_batch_as_is),
        DataLoader(val, batch_size=batch_size, collate_fn=_batch_as_is),
        DataLoader(test, batch_size=batch_size, collate_fn=_batch_as_is),
    )
    return loaders, y_normalizer
