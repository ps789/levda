"""Problem-independent interface for checkpoint-based LEVDA examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .assimilation import AssimilationConfig
from .networks import ResFourierLDNN, ResLDNN
from .normalization import GaussianNormalization, MinMaxNormalization


@dataclass(frozen=True)
class Problem:
    name: str
    folder: Path
    dynamics_sizes: tuple[int, ...]
    decoder_sizes: tuple[int, ...]
    assimilation: AssimilationConfig
    dynamics_checkpoint: str
    decoder_checkpoint: str
    observation_function: Callable
    fourier_size: int | None = None
    normalization_bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    default_num_inits: int = 1
    default_obs_sigma: float = 0.1
    initialization_seed_offset: int = 0
    seed_before_model: bool = False
    finite_initialization_averages: bool = False
    report_member_rms: bool = False

    def build_model(self, device):
        sizes = (list(self.dynamics_sizes), list(self.decoder_sizes))
        if self.fourier_size is None:
            return ResLDNN(*sizes).to(device)
        return ResFourierLDNN(self.fourier_size, *sizes).to(device)

    def load_normalization(self, checkpoint_dir=None):
        if self.normalization_bounds is not None:
            return MinMaxNormalization(*self.normalization_bounds)
        folder = Path(checkpoint_dir) if checkpoint_dir is not None else self.folder / "saved_model"
        return GaussianNormalization.from_file(folder / "mean_std.npz")
