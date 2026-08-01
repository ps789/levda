import numpy as np


class Normalize:
    def __init__(self, v_min, v_max):
        self.v_min = np.array(v_min, dtype=np.float32)
        self.v_max = np.array(v_max, dtype=np.float32)

    def normalize_forw(self, v):
        return (2.0 * v - self.v_min - self.v_max) / (self.v_max - self.v_min)

    def normalize_back(self, v):
        return 0.5 * (self.v_min + self.v_max + (self.v_max - self.v_min) * v)

