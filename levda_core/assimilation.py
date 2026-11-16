"""Shared nonlinear ensemble-space smoother with source-specific compatibility settings."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class AssimilationConfig:
    grid_shape: tuple[int, int]
    observation_stride: int
    ensemble_size: int = 20
    smoothing_steps: int = 5
    max_iter: int = 10
    learning_rate: float = 0.1
    parameter_offset: float = 1.0
    initial_parameter_noise: float = 0.2
    state_inflation: float = 0.0007
    parameter_inflation: float = 0.001
    observation_weight: float = 100.0
    current_observation_uses_background: bool = True
    gradient_clip: float | None = None
    clear_model_grads_each_closure: bool = False


def step_dt(dt, index):
    if not torch.is_tensor(dt) or dt.ndim == 0:
        return dt
    if dt.ndim == 1:
        return dt[index]
    if dt.ndim == 2:
        return dt[:, index:index + 1]
    raise ValueError(f"Unsupported dt shape: {tuple(dt.shape)}")


def assimilate(model, data, observations, config, *, device,
               output_ensemble=True, decode_chunk_size=4096,
               equilibrium=False, latent_init=False):
    x, dt, u = data["x"], data["dt"], data["u"]
    if u.ndim != 2 or u.shape[0] != 1 or x.shape[0] != 1:
        raise ValueError("Assimilate one trajectory at a time: u must have shape (1, parameters).")
    if config.ensemble_size < 2 or config.max_iter < 1 or config.smoothing_steps < 0:
        raise ValueError("Need ensemble_size >= 2, max_iter >= 1 and smoothing_steps >= 0.")
    e, latent_dim = config.ensemble_size, model.num_latent_states
    model.ensemble_size = e
    u = torch.repeat_interleave(u + config.parameter_offset, e, dim=0)
    u = u + torch.randn_like(u) * config.initial_parameter_noise
    model.state = data["latent"] if latent_init else torch.zeros(e, latent_dim, device=device)
    model.state_history, model.parameter_history = [], []
    grid = x[:, 0].reshape(1, *config.grid_shape, 2)
    coordinates = grid[:, ::config.observation_stride, ::config.observation_stride].reshape(1, -1, 2)
    n_obs = coordinates.shape[1]
    coordinates = torch.repeat_interleave(coordinates, e, dim=0)
    if observations.shape[1] != x.shape[1] or observations[0, 0].numel() != n_obs * model.rec.rec.hidden_output_linear.out_features:
        raise ValueError("Observations do not match the configured grid, channels, or time steps.")
    target = observations.reshape(1, x.shape[1], n_obs, -1)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    reconstructions = []
    scale = math.sqrt(e - 1)

    for i in range(x.shape[1]):
        with torch.no_grad():
            inputs = torch.cat((u, model.state), dim=1)
            velocity = model.dyn(inputs)
            if equilibrium:
                velocity = velocity - model.dyn(torch.zeros_like(inputs))
            model.state = model.state + step_dt(dt, i) * velocity

        u_i = torch.Tensor(u.clone()).to(device)
        inflated = torch.Tensor(model.state.clone() + config.state_inflation * torch.randn_like(model.state)).to(device)
        delta_u = torch.autograd.Variable(
            config.parameter_inflation * torch.randn_like(u_i), requires_grad=True
        )
        alpha = torch.autograd.Variable(torch.eye(e, device=device) * scale, requires_grad=True)
        # Both original Fourier implementations keep this clone attached to alpha.
        # Deliberately preserve that gradient behavior for checkpoint/result parity.
        alpha_start = alpha.clone()
        state_mean = torch.Tensor(torch.mean(inflated, dim=0, keepdim=True)).to(device)
        basis = ((inflated - state_mean) / scale).permute(1, 0)
        optimizer = torch.optim.LBFGS([alpha, delta_u], max_iter=config.max_iter,
                                      lr=config.learning_rate)

        def closure():
            optimizer.zero_grad()
            if config.clear_model_grads_each_closure:
                model.zero_grad(set_to_none=True)
            candidate = state_mean + torch.matmul(basis, alpha).permute(1, 0)
            # Kolmogorov's source uses the forecast for the current-time term;
            # PlanetSWE uses the candidate. Future terms depend on the candidate in both.
            current = model.state if config.current_observation_uses_background else candidate
            predicted = model.rec(coordinates, current.unsqueeze(1).expand(e, n_obs, latent_dim))
            loss = config.observation_weight * loss_fn(predicted, target[:, i].expand(e, -1, -1))
            for j in range(config.smoothing_steps):
                if i + j + 1 < x.shape[1]:
                    velocity = model.dyn(torch.cat((u_i + delta_u, candidate), dim=1))
                    candidate = candidate + step_dt(dt, i + j) * velocity
                    predicted = model.rec(coordinates, candidate.unsqueeze(1).expand(e, n_obs, latent_dim))
                    loss = loss + config.observation_weight * loss_fn(
                        predicted, target[:, i + j + 1].expand(e, -1, -1)
                    )
            gram = torch.matmul(alpha.permute(1, 0) - alpha_start.permute(1, 0), alpha - alpha_start)
            loss = loss + loss_fn(gram, torch.zeros_like(alpha))
            loss = loss + loss_fn(delta_u, torch.zeros_like(delta_u))
            loss.backward()
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_([alpha, delta_u], config.gradient_clip)
            return loss

        optimizer.step(closure)
        model.zero_grad(set_to_none=True)
        u = u_i + delta_u
        model.state = (state_mean + torch.matmul(basis, alpha).permute(1, 0)).detach()
        model.state_history.append(model.state)
        model.parameter_history.append(u.detach())
        with torch.no_grad():
            if output_ensemble:
                decode_state = model.state
                decode_coordinates = torch.repeat_interleave(x[:, 0], e, dim=0)
            else:
                decode_state = torch.mean(model.state, dim=0, keepdim=True)
                decode_coordinates = x[:, 0]
            reconstructions.append(model.decode(decode_coordinates, decode_state, decode_chunk_size).cpu())

    model.parameter_history = torch.stack(model.parameter_history, dim=1)
    return torch.stack(reconstructions, dim=1)
