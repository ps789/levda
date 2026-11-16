"""PlanetSWE architecture, observations, and published-run settings."""

from pathlib import Path

import torch

from levda_core.assimilation import AssimilationConfig
from levda_core.problem import Problem


def prepare_observations(data, config, device, noise):
    y = data["y"].to(device)
    grid = y.reshape(y.shape[0], y.shape[1], *config.grid_shape, -1)
    observations = grid[:, :, ::config.observation_stride, ::config.observation_stride]
    return observations + noise * torch.randn_like(observations) * observations


PROBLEM = Problem(
    name="planetswe",
    folder=Path(__file__).resolve().parent,
    fourier_size=50,
    dynamics_sizes=(52,) + (200,) * 8 + (50,),
    decoder_sizes=(52,) + (500,) * 15 + (3,),
    assimilation=AssimilationConfig(
        (512, 256), 32, learning_rate=0.001,
        parameter_offset=0.2, initial_parameter_noise=0.05,
        state_inflation=0.006, observation_weight=1.0,
        current_observation_uses_background=False, gradient_clip=1.0,
        clear_model_grads_each_closure=True,
    ),
    dynamics_checkpoint="dyn_1799.ckpt",
    decoder_checkpoint="rec_1799.ckpt",
    observation_function=prepare_observations,
    report_member_rms=True,
)
