"""Physical-space normalization used by the released checkpoints."""

import numpy as np


class GaussianNormalization:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    @classmethod
    def from_file(cls, path):
        with np.load(path, allow_pickle=False) as stats:
            return cls(stats["mean"], stats["std"])

    def normalize_back(self, values):
        return values * self.std + self.mean


class MinMaxNormalization:
    def __init__(self, minimum, maximum):
        self.minimum = np.asarray(minimum, dtype=np.float32)
        self.maximum = np.asarray(maximum, dtype=np.float32)

    def normalize_back(self, values):
        return 0.5 * (self.minimum + self.maximum + (self.maximum - self.minimum) * values)
