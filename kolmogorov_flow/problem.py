"""Kolmogorov-flow architecture, observations, and published-run settings."""

from pathlib import Path

import torch

from levda_core.assimilation import AssimilationConfig
from levda_core.problem import Problem


def prepare_observations(data, config, device, noise):
    # The original experiment draws noise on the full CPU grid before subsampling.
    y = data["y"]
    y = y + torch.randn_like(y) * noise * y
    grid = y.reshape(y.shape[0], y.shape[1], *config.grid_shape, -1)
    return grid[:, :, ::config.observation_stride, ::config.observation_stride].to(device)


PROBLEM = Problem(
    name="kolmogorov_flow",
    folder=Path(__file__).resolve().parent,
    fourier_size=10,
    dynamics_sizes=(10,) + (200,) * 9 + (9,),
    decoder_sizes=(11,) + (500,) * 14 + (2,),
    assimilation=AssimilationConfig((150, 150), 15),
    dynamics_checkpoint="dyn_1999.ckpt",
    decoder_checkpoint="retrained_rec_1999.ckpt",
    observation_function=prepare_observations,
)
