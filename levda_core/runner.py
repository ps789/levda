"""Small command-line interface shared by the two Fourier-LDNet examples."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import random
import time

import numpy as np
import torch

from .assimilation import assimilate
from .metrics import evaluate_ensemble
from .problems import PROBLEMS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path, max_steps=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    data = {key: payload["data_test"][key].contiguous() for key in ("x", "u", "y", "dt")}
    if max_steps is not None:
        if max_steps < 1 or max_steps > data["y"].shape[1]:
            raise ValueError("max_steps must be between 1 and the trajectory length.")
        for key in ("x", "y"):
            data[key] = data[key][:, :max_steps]
        if data["u"].ndim == 3:
            data["u"] = data["u"][:, :max_steps]
        if data["dt"].ndim == 1:
            data["dt"] = data["dt"][:max_steps]
    return data


def load_model(problem, device, checkpoint_dir=None):
    model = problem.build_model(device)
    folder = Path(checkpoint_dir) if checkpoint_dir else problem.folder / "saved_model"
    model.dyn.load_state_dict(torch.load(folder / problem.dynamics_checkpoint,
                                        map_location=device, weights_only=True), strict=True)
    model.rec.load_state_dict(torch.load(folder / problem.decoder_checkpoint,
                                        map_location=device, weights_only=True), strict=True)
    model.eval()
    return model


def prepare_observations(data, problem, device, noise=0.1):
    y = data["y"]
    config = problem.assimilation
    if problem.noise_on_model_device:
        y = y.to(device)
    if problem.noise_before_subsampling:
        # Preserve the original full-grid random draw before selecting the sensor grid.
        y = y + torch.randn_like(y) * noise * y
    obs = y.reshape(y.shape[0], y.shape[1], *config.grid_shape, -1)
    obs = obs[:, :, ::config.observation_stride, ::config.observation_stride, :]
    if not problem.noise_before_subsampling:
        obs = obs + noise * torch.randn_like(obs) * obs
    return obs.to(device)


def run(problem_name, argv=None):
    problem = PROBLEMS[problem_name]
    parser = argparse.ArgumentParser(description=f"Self-contained LEVDA: {problem_name}")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-path", type=Path, default=problem.folder / "data/sample_data.pth")
    parser.add_argument("--checkpoint-dir", type=Path, default=problem.folder / "saved_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads.")
    parser.add_argument("--num-inits", type=int, default=1)
    parser.add_argument("--obs-sigma", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, help="Shorter smoke test; omitted runs the full trajectory.")
    parser.add_argument("--ensemble-size", type=int, default=problem.assimilation.ensemble_size)
    parser.add_argument("--max-iter", type=int, default=problem.assimilation.max_iter)
    parser.add_argument("--learning-rate", type=float, default=problem.assimilation.learning_rate)
    parser.add_argument("--decode-chunk-size", type=int, default=4096,
                        help="Physical-grid points per decoder batch; 0 reproduces monolithic decoding.")
    parser.add_argument("--print-curves", action="store_true")
    args = parser.parse_args(argv)
    if args.num_inits < 1 or args.obs_sigma < 0 or args.decode_chunk_size < 0 or args.threads < 1:
        parser.error("Need num-inits >= 1, obs-sigma >= 0, decode-chunk-size >= 0, threads >= 1.")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    data = load_data(args.data_path, args.max_steps)
    model = load_model(problem, device, args.checkpoint_dir)
    stats = np.load(args.checkpoint_dir / "mean_std.npz", allow_pickle=False)
    truth = data["y"].numpy() * stats["std"] + stats["mean"]
    config = replace(problem.assimilation, ensemble_size=args.ensemble_size,
                     max_iter=args.max_iter, learning_rate=args.learning_rate)
    model_data = {key: data[key].to(device) for key in ("x", "u", "dt")}
    set_seed(args.seed)
    observations = prepare_observations(data, problem, device, args.obs_sigma)
    print(f"{problem_name}: {data['y'].shape[1]} steps, {config.ensemble_size} members, "
          f"{config.max_iter} L-BFGS iterations, lr={config.learning_rate}, device={device}", flush=True)
    trials = []
    for i in range(args.num_inits):
        if i:
            set_seed(args.seed + i)
        start = time.perf_counter()
        outputs = assimilate(model, model_data, observations, config, device=device,
                              decode_chunk_size=args.decode_chunk_size)
        physical = outputs.numpy() * stats["std"] + stats["mean"]
        metrics = evaluate_ensemble(physical, truth)
        trials.append(metrics)
        print(f"initialization {i + 1}: mean relative RMSE={np.mean(metrics['relative_rmse']):.8f}, "
              f"final={metrics['relative_rmse'][-1]:.8f}, mean SER={np.mean(metrics['ser']):.8f}, "
              f"time={time.perf_counter() - start:.2f}s", flush=True)
        if problem_name == "planetswe":
            print(f"  original-script member-RMS relative RMSE={np.mean(metrics['member_rms_relative_rmse']):.8f}", flush=True)
    averages = {key: np.mean([trial[key] for trial in trials], axis=0) for key in trials[0]}
    print(f"Average relative RMSE over {args.num_inits} initialization(s): {np.mean(averages['relative_rmse']):.8f}")
    print(f"Average SER over {args.num_inits} initialization(s): {np.mean(averages['ser']):.8f}")
    if args.print_curves:
        for key, values in averages.items():
            print(f"{key}: {np.array2string(values, precision=8, threshold=10000)}")
    return averages
