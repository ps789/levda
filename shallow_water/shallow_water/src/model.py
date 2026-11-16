import math

import numpy as np
import torch
import torch.nn as nn


mseloss = nn.MSELoss(reduction="sum")


def _step_dt(dt, step_idx):
    if not torch.is_tensor(dt):
        return dt
    if dt.ndim == 0:
        return dt
    if dt.ndim == 1:
        return dt[step_idx]
    if dt.ndim == 2:
        return dt[:, step_idx].unsqueeze(1)
    raise ValueError(f"Unsupported dt shape: {tuple(dt.shape)}")


class LDNN(nn.Module):
    def __init__(self, dyn, rec):
        super().__init__()
        self.dyn = dyn
        self.rec = rec
        self.num_latent_states = dyn.layers[-1].out_features
        self.state = None
        self.state_history = []

    def forward(self, data, device, equilibrium=False, latent_state=False, latent_init=False):
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]

        self.state = data["latent"] if latent_init else torch.zeros(u.shape[0], self.num_latent_states, device=device)
        self.state_history = []

        for i in range(x.shape[1]):
            u_ti = u if u.ndim == 2 else u[:, i, :]
            input_ = torch.cat((u_ti, self.state), dim=1)
            step_dt = _step_dt(dt, i)
            if equilibrium:
                self.state = self.state + step_dt * (self.dyn(input_) - self.dyn(torch.zeros_like(input_)))
            else:
                self.state = self.state + step_dt * self.dyn(input_)
            self.state_history.append(self.state)

        self.state_history = torch.stack(self.state_history).transpose(0, 1)

        if latent_state:
            return self.state_history

        ni, nt, nx, dims = (
            self.state_history.shape[0],
            self.state_history.shape[1],
            x.shape[2],
            self.state_history.shape[2],
        )
        state_expanded = self.state_history.unsqueeze(2).expand(ni, nt, nx, dims)
        if self.training:
            return self.rec(torch.cat([x, state_expanded], dim=-1))
        with torch.no_grad():
            outputs = []
            x_rec_input = torch.cat([x, state_expanded], dim=-1).detach()
            for start in np.arange(0, self.ensemble_size, 5):
                outputs.append(self.rec(x_rec_input[start : start + 5]).detach())
            return torch.cat(outputs, dim=0)

    def rec_combined(self, obs_indices, state):
        return self.rec(torch.cat([obs_indices, state], dim=-1))

    def forward_data_assimilation_lbfgs_with_dynamics_multistep_ensemble_cov(
        self,
        data,
        observations,
        device,
        equilibrium=False,
        latent_state=False,
        latent_init=False,
        output_ensemble=False,
        obs_R=1,
        alpha_reg=0.0,
    ):
        del latent_state

        smoothing_steps = 5
        u = data["u"][:, 0, :]
        self.ensemble_size = 30
        u = u - 0.1
        u = torch.repeat_interleave(u, self.ensemble_size, 0)
        u = u + torch.randn_like(u) * 0.1
        x = data["x"]
        dt = data["dt"]
        x_rec = []
        u_hist = []

        assert u.ndim == 2

        self.state = data["latent"] if latent_init else torch.zeros(self.ensemble_size, self.num_latent_states, device=device)
        self.state_history = []

        observation_indices = x[:, 0, :, :].reshape(x.shape[0], 150, 150, 2)[:, ::15, ::15, :]
        obs_dim = observation_indices.reshape(observation_indices.shape[0], -1, 2).shape[1]
        observation_indices = torch.repeat_interleave(observation_indices, self.ensemble_size, 0)
        observation_indices = observation_indices.reshape(self.ensemble_size, obs_dim, 2)

        lambda_u = 1.0

        for i in range(x.shape[1]):
            u_hist.append(u)
            with torch.no_grad():
                u_ti = u if u.ndim == 2 else u[:, i, :]
                input_ = torch.cat((u_ti, self.state), dim=1)
                step_dt = _step_dt(dt, i)
                if equilibrium:
                    self.state = self.state + step_dt * (self.dyn(input_) - self.dyn(torch.zeros_like(input_)))
                else:
                    self.state = self.state + step_dt * self.dyn(input_)

            if i < x.shape[1] - smoothing_steps:
                u_ti = (u if u.ndim == 2 else u[:, i, :]).clone().to(device)
                state = (self.state.clone() + 0.0006 * torch.randn_like(self.state)).to(device)
                u_ti_perturb = torch.autograd.Variable(0.0001 * torch.randn_like(u_ti).to(device), requires_grad=True)
                alpha = torch.autograd.Variable(
                    torch.eye(self.ensemble_size, device=device) * math.sqrt(self.ensemble_size - 1),
                    requires_grad=True,
                )
                max_iter = 40
                state_mean = torch.mean(state, dim=0, keepdim=True)
                perturb_basis = state - state_mean
                Px = (perturb_basis / math.sqrt(self.ensemble_size - 1)).permute(1, 0)

                def loss_closure():
                    opt.zero_grad()
                    temp_states = [state_mean + torch.matmul(Px, alpha).permute(1, 0)]
                    state_var_ = self.state.unsqueeze(1).expand(self.ensemble_size, obs_dim, self.num_latent_states)
                    x_rec_i = self.rec_combined(observation_indices, state_var_)
                    target_obs = observations[:, i, :, :, :].reshape(1, obs_dim, 3).expand(self.ensemble_size, obs_dim, 3)
                    loss = mseloss(x_rec_i, target_obs) * obs_R

                    for j in range(smoothing_steps):
                        step_dt = _step_dt(dt, i + j)
                        dyn_output = self.dyn(torch.cat((u_ti + u_ti_perturb, temp_states[j]), dim=1))
                        temp_states.append(temp_states[j] + step_dt * dyn_output)
                        state_var_ = temp_states[-1].unsqueeze(1).expand(self.ensemble_size, obs_dim, self.num_latent_states)
                        x_rec_i = self.rec_combined(observation_indices, state_var_)
                        target_obs = observations[:, i + j + 1, :, :, :].reshape(1, obs_dim, 3).expand(self.ensemble_size, obs_dim, 3)
                        loss = loss + mseloss(x_rec_i, target_obs) * obs_R

                    if alpha_reg > 0:
                        alpha_delta = alpha - alpha_start
                        loss = loss + torch.mean(torch.matmul(alpha_delta, alpha_delta.T) ** 2) * alpha_reg
                    loss = loss + mseloss(u_ti_perturb, torch.zeros_like(u_ti_perturb)) * lambda_u
                    loss.backward()
                    return loss

                opt = torch.optim.LBFGS([alpha, u_ti_perturb], max_iter=max_iter, lr=0.1, line_search_fn="strong_wolfe")
                alpha_start = alpha.detach().clone()
                opt.step(loss_closure)

                for var in self.rec.parameters():
                    var.grad = None
                for var in self.dyn.parameters():
                    var.grad = None

                u = u_ti + u_ti_perturb
                self.state = (state_mean + torch.matmul(Px, alpha).permute(1, 0)).detach()
                self.state_history.append(self.state)

            if output_ensemble:
                state_var_ = self.state.unsqueeze(1).expand(self.ensemble_size, x.shape[2], self.num_latent_states)
            else:
                state_var_ = torch.mean(self.state, dim=0, keepdim=True).unsqueeze(1).expand(1, x.shape[2], self.num_latent_states)

            with torch.no_grad():
                if output_ensemble:
                    x_rec_i = self.rec_combined(torch.repeat_interleave(x, self.ensemble_size, 0)[:, 0, :, :], state_var_).detach().cpu()
                else:
                    x_rec_i = self.rec_combined(x[:, 0, :, :], state_var_).detach().cpu()
                x_rec.append(x_rec_i)

        x_rec_tensor = torch.stack(x_rec, dim=1)
        u_hist_tensor = torch.stack(u_hist, dim=1)
        return x_rec_tensor, u_hist_tensor
