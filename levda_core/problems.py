"""Architecture and assimilation presets copied from the original entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .assimilation import AssimilationConfig
from .networks import ResFourierLDNN


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Problem:
    name: str
    fourier_size: int
    dynamics_sizes: tuple[int, ...]
    decoder_sizes: tuple[int, ...]
    assimilation: AssimilationConfig
    dynamics_checkpoint: str
    decoder_checkpoint: str
    noise_before_subsampling: bool
    noise_on_model_device: bool

    @property
    def folder(self):
        return ROOT / self.name

    def build_model(self, device):
        return ResFourierLDNN(self.fourier_size, list(self.dynamics_sizes),
                              list(self.decoder_sizes)).to(device)


PROBLEMS = {
    "kolmogorov_flow": Problem(
        "kolmogorov_flow", 10, (10,) + (200,) * 9 + (9,),
        (11,) + (500,) * 14 + (2,), AssimilationConfig((150, 150), 15),
        "dyn_1999.ckpt", "retrained_rec_1999.ckpt", True, False,
    ),
    "planetswe": Problem(
        "planetswe", 50, (52,) + (200,) * 8 + (50,),
        (52,) + (500,) * 15 + (3,),
        AssimilationConfig((512, 256), 32, learning_rate=0.001,
                           parameter_offset=0.2, initial_parameter_noise=0.05,
                           state_inflation=0.006, observation_weight=1.0,
                           current_observation_uses_background=False, gradient_clip=1.0,
                           clear_model_grads_each_closure=True),
        "dyn_1799.ckpt", "rec_1799.ckpt", False, True,
    ),
}
