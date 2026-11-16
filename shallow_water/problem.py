"""Tsunami architecture, observations, and original ensemble-L-BFGS settings."""

from pathlib import Path

import torch

from levda_core.assimilation import AssimilationConfig
from levda_core.problem import Problem


def prepare_observations(data, config, device, noise):
    y = data["y"]
    grid = y.clone().reshape(y.shape[0], y.shape[1], *config.grid_shape, -1)
    observations = grid[:, :, ::config.observation_stride, ::config.observation_stride].to(device)
    if noise > 0:
        observations = observations + observations * noise * torch.randn_like(observations)
    return observations


PROBLEM = Problem(
    name="shallow_water",
    folder=Path(__file__).resolve().parent,
    dynamics_sizes=(12,) + (50,) * 8 + (10,),
    decoder_sizes=(12,) + (300,) * 10 + (3,),
    assimilation=AssimilationConfig(
        (150, 150), 15, ensemble_size=30, max_iter=40,
        parameter_offset=-0.1, initial_parameter_noise=0.1,
        state_inflation=0.0006, parameter_inflation=0.0001,
        observation_weight=1.0, line_search_fn="strong_wolfe",
        assimilate_incomplete_window=False, alpha_regularization="increment_row_gram",
        tensor_constructor_aliases=False,
    ),
    dynamics_checkpoint="dyn_1999.ckpt",
    decoder_checkpoint="retrained_rec_1999.ckpt",
    observation_function=prepare_observations,
    normalization_bounds=((-0.2, -0.2, -0.5), (0.2, 0.2, 1.0)),
    default_num_inits=5,
    default_obs_sigma=0.05,
    initialization_seed_offset=1,
    seed_before_model=True,
    finite_initialization_averages=True,
)
