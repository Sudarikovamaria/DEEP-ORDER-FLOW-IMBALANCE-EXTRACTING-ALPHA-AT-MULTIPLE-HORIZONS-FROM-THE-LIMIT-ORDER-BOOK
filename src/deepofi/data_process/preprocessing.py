import numpy as np


class Normalizer:
    """Per-column winsorization and z-score, fitted on training data only (HANDOFF §7)."""

    def __init__(self, quantiles=(0.005, 0.995)):
        self.quantiles = quantiles

    def fit(self, arrays):
        data = np.concatenate([np.asarray(a, dtype=np.float64) for a in arrays])
        self.lower, self.upper = np.quantile(data, self.quantiles, axis=0)
        clipped = np.clip(data, self.lower, self.upper)
        self.mean = clipped.mean(axis=0)
        self.std = clipped.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, x):
        return ((np.clip(x, self.lower, self.upper) - self.mean) / self.std).astype(np.float32)

    def inverse(self, x):
        return x * self.std + self.mean
