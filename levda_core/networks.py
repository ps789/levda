"""Checkpoint-compatible LDNet components without a DeepXDE dependency."""

from __future__ import annotations

import torch
from torch import nn


class DenseNetwork(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.linears = nn.ModuleList()
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            layer = nn.Linear(n_in, n_out)
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = torch.relu(layer(x))
        return self.linears[-1](x)


class ActivationModule(nn.Module):
    def forward(self, x):
        return torch.relu(x)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        return torch.relu(self.linear2(torch.relu(self.linear1(x))) + self.shortcut(x))


class ResidualNetwork(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        hidden_depth, width = len(sizes) - 2, sizes[1]
        self.residual_blocks = nn.ModuleList(
            ResidualBlock(width) for _ in range((hidden_depth - 1) // 2)
        )
        self.remaining_hidden_layers = nn.ModuleList()
        if (hidden_depth - 1) % 2:
            self.remaining_hidden_layers.extend([nn.Linear(width, width), ActivationModule()])
        self.input_hidden_linear = nn.Linear(sizes[0], width)
        self.hidden_output_linear = nn.Linear(width, sizes[-1])
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = torch.relu(self.input_hidden_linear(x))
        for block in self.residual_blocks:
            x = block(x)
        for layer in self.remaining_hidden_layers:
            x = layer(x)
        return self.hidden_output_linear(x)


class FourierEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.encoding = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.encoding(x)


class FourierDecoder(nn.Module):
    def __init__(self, embedding, network):
        super().__init__()
        self.fourier_embedding = embedding
        self.rec = network

    def forward(self, coordinates, states):
        projected = self.fourier_embedding(2 * torch.pi * coordinates)
        return self.rec(torch.cat([torch.sin(projected), torch.cos(projected), states], dim=-1))


class ResFourierLDNN(nn.Module):
    def __init__(self, fourier_mapping_size, layer_sizes_dyn, layer_sizes_rec):
        super().__init__()
        self.num_latent_states = layer_sizes_dyn[-1]
        self.dyn = DenseNetwork(layer_sizes_dyn)
        self.B = FourierEmbedding(layer_sizes_rec[0] - self.num_latent_states, fourier_mapping_size)
        decoder_sizes = [self.num_latent_states + 2 * fourier_mapping_size] + layer_sizes_rec[1:]
        self.rec = FourierDecoder(self.B, ResidualNetwork(decoder_sizes))
        self.state = None
        self.state_history = []

    def decode(self, coordinates, state, chunk_size=4096):
        """Chunk only physical-space reconstruction, never the assimilation loss."""
        n_points = coordinates.shape[-2]
        if state.ndim == coordinates.ndim - 1:
            state = state.unsqueeze(-2).expand(*coordinates.shape[:-1], self.num_latent_states)
        if not chunk_size or chunk_size >= n_points:
            return self.rec(coordinates, state)
        return torch.cat([
            self.rec(coordinates[..., start:start + chunk_size, :],
                     state[..., start:start + chunk_size, :])
            for start in range(0, n_points, chunk_size)
        ], dim=-2)

    def forward(self, data, device=None, equilibrium=False, latent_state=False,
                latent_init=False, decode_chunk_size=4096):
        from .assimilation import step_dt

        device = device or next(self.parameters()).device
        u, x, dt = data["u"], data["x"], data["dt"]
        self.state = data["latent"] if latent_init else torch.zeros(
            u.shape[0], self.num_latent_states, device=device
        )
        history = []
        for i in range(x.shape[1]):
            u_i = u if u.ndim == 2 else u[:, i]
            inputs = torch.cat((u_i, self.state), dim=1)
            velocity = self.dyn(inputs)
            if equilibrium:
                velocity = velocity - self.dyn(torch.zeros_like(inputs))
            self.state = self.state + step_dt(dt, i) * velocity
            history.append(self.state)
        self.state_history = torch.stack(history, dim=1)
        if latent_state:
            return self.state_history
        return self.decode(x, self.state_history, decode_chunk_size)

    def forward_data_assimilation_lbfgs_with_dynamics_multistep_ensemble_cov(
        self, data, observations, device, *, config, output_ensemble=False,
        decode_chunk_size=4096, equilibrium=False, latent_init=False,
    ):
        from .assimilation import assimilate

        return assimilate(self, data, observations, config, device=device,
                          output_ensemble=output_ensemble,
                          decode_chunk_size=decode_chunk_size,
                          equilibrium=equilibrium, latent_init=latent_init)
