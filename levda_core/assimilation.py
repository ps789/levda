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
    line_search_fn: str | None = None
    assimilate_incomplete_window: bool = True
    alpha_regularization: str = "attached_column_gram"
    alpha_regularization_weight: float = 1.0
    tensor_constructor_aliases: bool = True


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
    if u.ndim == 3:
        u = u[:, 0]
    if u.ndim != 2 or u.shape[0] != 1 or x.shape[0] != 1:
        raise ValueError("Assimilate one trajectory at a time: u must have shape (1, parameters).")
    if config.ensemble_size < 2 or config.max_iter < 1 or config.smoothing_steps < 0:
        raise ValueError("Need ensemble_size >= 2, max_iter >= 1 and smoothing_steps >= 0.")
    if config.alpha_regularization not in ("attached_column_gram", "increment_row_gram"):
        raise ValueError("Unknown alpha regularization.")
    if config.alpha_regularization_weight < 0 or config.observation_weight < 0:
        raise ValueError("Regularization and observation weights must be nonnegative.")
    e, latent_dim = config.ensemble_size, model.num_latent_states
    model.ensemble_size = e
    u = torch.repeat_interleave(u + config.parameter_offset, e, dim=0)
    u = u + torch.randn_like(u) * config.initial_parameter_noise
    model.state = data["latent"] if latent_init else torch.zeros(e, latent_dim, device=device)
    model.state_history, model.parameter_history = [], []
    model.forecast_parameter_history, model.analysis_indices = [], []
    grid = x[:, 0].reshape(1, *config.grid_shape, 2)
    coordinates = grid[:, ::config.observation_stride, ::config.observation_stride].reshape(1, -1, 2)
    n_obs = coordinates.shape[1]
    coordinates = torch.repeat_interleave(coordinates, e, dim=0)
    if observations.shape[1] != x.shape[1] or observations[0, 0].numel() != n_obs * model.num_outputs:
        raise ValueError("Observations do not match the configured grid, channels, or time steps.")
    target = observations.reshape(1, x.shape[1], n_obs, -1)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    reconstructions = []
    scale = math.sqrt(e - 1)

    def source_tensor(value):
        # Retain the source's autograd graph, including legacy tensor aliases.
        return torch.Tensor(value).to(device) if config.tensor_constructor_aliases else value.to(device)

    for i in range(x.shape[1]):
        model.forecast_parameter_history.append(u.detach())
        with torch.no_grad():
            inputs = torch.cat((u, model.state), dim=1)
            velocity = model.dyn(inputs)
            if equilibrium:
                velocity = velocity - model.dyn(torch.zeros_like(inputs))
            model.state = model.state + step_dt(dt, i) * velocity

        if config.assimilate_incomplete_window or i < x.shape[1] - config.smoothing_steps:
            u_i = source_tensor(u.clone())
            inflated = source_tensor(model.state.clone() + config.state_inflation * torch.randn_like(model.state))
            delta_u = torch.autograd.Variable(
                config.parameter_inflation * torch.randn_like(u_i), requires_grad=True
            )
            alpha = torch.autograd.Variable(torch.eye(e, device=device) * scale, requires_grad=True)
            # The Fourier sources keep this clone attached; tsunami regularizes
            # a real increment from the detached initial ensemble transform.
            alpha_start = alpha.clone() if config.alpha_regularization == "attached_column_gram" else alpha.detach().clone()
            state_mean = source_tensor(torch.mean(inflated, dim=0, keepdim=True))
            basis = ((inflated - state_mean) / scale).permute(1, 0)
            optimizer = torch.optim.LBFGS([alpha, delta_u], max_iter=config.max_iter,
                                          lr=config.learning_rate, line_search_fn=config.line_search_fn)

            def closure():
                optimizer.zero_grad()
                if config.clear_model_grads_each_closure:
                    model.zero_grad(set_to_none=True)
                candidate = state_mean + torch.matmul(basis, alpha).permute(1, 0)
                current = model.state if config.current_observation_uses_background else candidate
                predicted = model.reconstruct(coordinates, current.unsqueeze(1).expand(e, n_obs, latent_dim))
                loss = config.observation_weight * loss_fn(predicted, target[:, i].expand(e, -1, -1))
                for j in range(config.smoothing_steps):
                    if i + j + 1 < x.shape[1]:
                        velocity = model.dyn(torch.cat((u_i + delta_u, candidate), dim=1))
                        candidate = candidate + step_dt(dt, i + j) * velocity
                        predicted = model.reconstruct(coordinates, candidate.unsqueeze(1).expand(e, n_obs, latent_dim))
                        loss = loss + config.observation_weight * loss_fn(
                            predicted, target[:, i + j + 1].expand(e, -1, -1)
                        )
                if config.alpha_regularization_weight > 0:
                    if config.alpha_regularization == "attached_column_gram":
                        gram = torch.matmul(alpha.permute(1, 0) - alpha_start.permute(1, 0), alpha - alpha_start)
                        penalty = loss_fn(gram, torch.zeros_like(alpha))
                    else:
                        increment = alpha - alpha_start
                        penalty = torch.mean(torch.matmul(increment, increment.T) ** 2)
                    loss = loss + penalty * config.alpha_regularization_weight
                loss = loss + loss_fn(delta_u, torch.zeros_like(delta_u))
                loss.backward()
                if config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_([alpha, delta_u], config.gradient_clip)
                return loss

            optimizer.step(closure)
            model.zero_grad(set_to_none=True)
            u = u_i + delta_u
            model.state = (state_mean + torch.matmul(basis, alpha).permute(1, 0)).detach()
            model.analysis_indices.append(i)
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
    model.forecast_parameter_history = torch.stack(model.forecast_parameter_history, dim=1)
    return torch.stack(reconstructions, dim=1)
