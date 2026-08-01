from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src.normalization import Normalize
from src.resnet import ResLDNN


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "sample_data.pth"
DEFAULT_DYN_CKPT = BASE_DIR / "saved_model" / "coriolis_recd_10_w_300_lr_1e-3_bs_2_ic_nor_u" / "dyn_1999.ckpt"
DEFAULT_REC_CKPT = BASE_DIR / "saved_model" / "coriolis_recd_10_w_300_lr_1e-3_bs_2_ic_nor_u" / "retrained_rec_1999.ckpt"


def pick_device(device_str: str | None):
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    return torch.device("cpu")


def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model, dyn_ckpt: Path, rec_ckpt: Path, device):
    model.dyn.load_state_dict(torch.load(dyn_ckpt, map_location=device))
    model.rec.load_state_dict(torch.load(rec_ckpt, map_location=device))
    return model


def prepare_observation(y: torch.Tensor, obs_sigma: float, device):
    observation = y.clone().reshape(y.shape[0], y.shape[1], 150, 150, 3)[:, :, ::15, ::15, :].to(device)
    if obs_sigma and obs_sigma > 0:
        observation = observation + observation * obs_sigma * torch.randn_like(observation)
    return observation


def compute_ser_simple(ens: np.ndarray, truth: np.ndarray, eps: float = 1e-12):
    mean_ens = np.mean(ens, axis=0)
    std_ens = np.std(ens, axis=0)
    err = np.sqrt(np.mean((mean_ens - truth[0]) ** 2, axis=(1, 2)))
    spread = np.sqrt(np.mean(std_ens**2, axis=(1, 2)))
    return spread / (err + eps)


def compute_time_slice_error_assimilate(data, model, device, normalize_y=None, num_inits=5, obs_sigma=0.05, obs_R=1, alpha_reg=1, seed=42):
    truth = data["y"][:1].cpu().numpy()
    if normalize_y is not None:
        truth = normalize_y.normalize_back(truth)

    observation = prepare_observation(data["y"][:1], obs_sigma, device)

    trial_errors = []
    trial_ser = []
    trial_error_means = []
    trial_ser_means = []

    for init_idx in range(num_inits):
        set_seed(None if seed is None else seed + init_idx + 1)
        data_test = {
            "x": data["x"].to(device),
            "u": data["u"].to(device),
            "dt": data["dt"].to(device),
        }
        outputs, _ = model.forward_data_assimilation_lbfgs_with_dynamics_multistep_ensemble_cov(
            data_test,
            observation.clone(),
            device=device,
            output_ensemble=True,
            obs_R=obs_R,
            alpha_reg=alpha_reg,
        )
        outputs = outputs.detach().cpu().numpy()
        if normalize_y is not None:
            outputs = normalize_y.normalize_back(outputs)

        prediction = np.mean(outputs, axis=0, keepdims=True)
        numerator = np.sum((prediction - truth) ** 2, axis=(0, 2, 3))
        denominator = np.sum(truth**2, axis=(0, 2, 3))
        error = np.sqrt(numerator / denominator)
        ser = compute_ser_simple(outputs, truth)

        trial_errors.append(error)
        trial_ser.append(ser)
        trial_error_means.append(np.mean(error))
        trial_ser_means.append(np.mean(ser))

    def _nanmean(values, axis=0):
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.nan
        totals = np.where(finite, values, 0.0).sum(axis=axis)
        counts = finite.sum(axis=axis)
        return np.divide(totals, counts, out=np.full_like(totals, np.nan, dtype=np.float64), where=counts > 0)

    error = _nanmean(trial_errors, axis=0)
    ser = _nanmean(trial_ser, axis=0)
    av_error = float(_nanmean(trial_error_means, axis=0))
    av_ser = float(_nanmean(trial_ser_means, axis=0))

    finite_rmse_runs = np.isfinite(trial_error_means).sum()
    finite_ser_runs = np.isfinite(trial_ser_means).sum()
    print(f"average relative RMSE over {num_inits} initializations ({finite_rmse_runs} finite): {av_error:.6f}")
    print(f"average SER over {num_inits} initializations ({finite_ser_runs} finite): {av_ser:.6f}")
    return error, av_error, ser, av_ser


def build_model(device):
    model = ResLDNN(
        [2 + 10] + 8 * [50] + [10],
        [2 + 10] + 10 * [300] + [3],
        activation="relu",
        kernel_initializer="Glorot normal",
    )
    model.to(device)
    return model


def load_sample_data(data_path: Path):
    payload = torch.load(data_path, map_location="cpu", weights_only=True)
    data = payload["data_test"]
    return {
        "x": data["x"].contiguous(),
        "u": data["u"].contiguous(),
        "y": data["y"].contiguous(),
        "dt": data["dt"].contiguous(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--dyn-ckpt", type=Path, default=DEFAULT_DYN_CKPT)
    parser.add_argument("--rec-ckpt", type=Path, default=DEFAULT_REC_CKPT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-inits", type=int, default=5)
    parser.add_argument("--obs-sigma", type=float, default=0.05)
    parser.add_argument("--obs-r", type=float, default=1.0)
    parser.add_argument("--alpha-reg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = pick_device(args.device)
    set_seed(args.seed)
    data = load_sample_data(args.data_path)
    model = build_model(device)
    load_model(model, args.dyn_ckpt, args.rec_ckpt, device)
    normalize_y = Normalize([-0.2, -0.2, -0.5], [0.2, 0.2, 1.0])
    compute_time_slice_error_assimilate(
        data,
        model,
        device,
        normalize_y,
        num_inits=args.num_inits,
        obs_sigma=args.obs_sigma,
        obs_R=args.obs_r,
        alpha_reg=args.alpha_reg,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
