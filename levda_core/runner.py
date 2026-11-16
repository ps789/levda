"""Problem-independent command-line runner for the LEVDA examples."""

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


def load_model(problem, device, checkpoint_dir=None, dyn_ckpt=None, rec_ckpt=None):
    model = problem.build_model(device)
    folder = Path(checkpoint_dir) if checkpoint_dir else problem.folder / "saved_model"
    model.dyn.load_state_dict(torch.load(dyn_ckpt or folder / problem.dynamics_checkpoint,
                                        map_location=device, weights_only=True), strict=True)
    model.rec.load_state_dict(torch.load(rec_ckpt or folder / problem.decoder_checkpoint,
                                        map_location=device, weights_only=True), strict=True)
    model.eval()
    return model


def prepare_observations(data, problem, device, noise=0.1):
    return problem.observation_function(data, problem.assimilation, device, noise)


def average(values, *, finite_only=False, axis=0):
    if not finite_only:
        return np.mean(values, axis=axis)
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    totals = np.where(finite, values, 0.0).sum(axis=axis)
    counts = finite.sum(axis=axis)
    return np.divide(totals, counts, out=np.full_like(totals, np.nan, dtype=np.float64), where=counts > 0)


def run(problem, argv=None):
    parser = argparse.ArgumentParser(description=f"Self-contained LEVDA: {problem.name}")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-path", type=Path, default=problem.folder / "data/sample_data.pth")
    parser.add_argument("--checkpoint-dir", type=Path, default=problem.folder / "saved_model")
    parser.add_argument("--dyn-ckpt", type=Path, help="Override the dynamics checkpoint.")
    parser.add_argument("--rec-ckpt", type=Path, help="Override the decoder checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads.")
    parser.add_argument("--num-inits", type=int, default=problem.default_num_inits)
    parser.add_argument("--obs-sigma", type=float, default=problem.default_obs_sigma)
    parser.add_argument("--obs-r", type=float, default=problem.assimilation.observation_weight,
                        help="Observation-loss multiplier, not a variance.")
    parser.add_argument("--alpha-reg", type=float, default=problem.assimilation.alpha_regularization_weight)
    parser.add_argument("--max-steps", type=int, help="Shorter smoke test; omitted runs the full trajectory.")
    parser.add_argument("--ensemble-size", type=int, default=problem.assimilation.ensemble_size)
    parser.add_argument("--max-iter", type=int, default=problem.assimilation.max_iter)
    parser.add_argument("--learning-rate", type=float, default=problem.assimilation.learning_rate)
    parser.add_argument("--smoothing-steps", type=int, default=problem.assimilation.smoothing_steps)
    parser.add_argument("--decode-chunk-size", type=int, default=4096,
                        help="Physical-grid points per decoder batch; 0 reproduces monolithic decoding.")
    parser.add_argument("--print-curves", action="store_true")
    args = parser.parse_args(argv)
    if args.num_inits < 1 or args.obs_sigma < 0 or args.decode_chunk_size < 0 or args.threads < 1:
        parser.error("Need num-inits >= 1, obs-sigma >= 0, decode-chunk-size >= 0, threads >= 1.")
    if args.ensemble_size < 2 or args.max_iter < 1 or args.learning_rate <= 0 or args.smoothing_steps < 0:
        parser.error("Need ensemble-size >= 2, max-iter >= 1, learning-rate > 0, smoothing-steps >= 0.")
    if args.obs_r < 0 or args.alpha_reg < 0:
        parser.error("Need obs-r >= 0 and alpha-reg >= 0.")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    data = load_data(args.data_path, args.max_steps)
    if problem.seed_before_model:
        set_seed(args.seed)
    model = load_model(problem, device, args.checkpoint_dir, args.dyn_ckpt, args.rec_ckpt)
    normalization = problem.load_normalization(args.checkpoint_dir)
    truth = normalization.normalize_back(data["y"].numpy())
    config = replace(problem.assimilation, ensemble_size=args.ensemble_size,
                     max_iter=args.max_iter, learning_rate=args.learning_rate,
                     smoothing_steps=args.smoothing_steps, observation_weight=args.obs_r,
                     alpha_regularization_weight=args.alpha_reg)
    model_data = {key: data[key].to(device) for key in ("x", "u", "dt")}
    if not problem.seed_before_model:
        set_seed(args.seed)
    observations = prepare_observations(data, problem, device, args.obs_sigma)
    print(f"{problem.name}: {data['y'].shape[1]} steps, {config.ensemble_size} members, "
          f"{config.max_iter} L-BFGS iterations, lr={config.learning_rate}, device={device}", flush=True)
    trials = []
    for i in range(args.num_inits):
        if i or problem.initialization_seed_offset:
            set_seed(args.seed + i + problem.initialization_seed_offset)
        start = time.perf_counter()
        outputs = assimilate(model, model_data, observations, config, device=device,
                              decode_chunk_size=args.decode_chunk_size)
        physical = normalization.normalize_back(outputs.numpy())
        metrics = evaluate_ensemble(physical, truth)
        trials.append(metrics)
        print(f"initialization {i + 1}: mean relative RMSE={np.mean(metrics['relative_rmse']):.8f}, "
              f"final={metrics['relative_rmse'][-1]:.8f}, mean SER={np.mean(metrics['ser']):.8f}, "
              f"time={time.perf_counter() - start:.2f}s", flush=True)
        if problem.report_member_rms:
            print(f"  original-script member-RMS relative RMSE={np.mean(metrics['member_rms_relative_rmse']):.8f}", flush=True)
    finite_only = problem.finite_initialization_averages
    averages = {key: average([trial[key] for trial in trials], finite_only=finite_only) for key in trials[0]}
    for key, label in (("relative_rmse", "relative RMSE"), ("ser", "SER")):
        trial_means = [np.mean(trial[key]) for trial in trials]
        overall = average(trial_means, finite_only=True) if finite_only else np.mean(averages[key])
        print(f"Average {label} over {args.num_inits} initialization(s): {overall:.8f}")
        if finite_only:
            print(f"  {np.isfinite(trial_means).sum()}/{args.num_inits} finite initialization means; no reruns")
    if args.print_curves:
        for key, values in averages.items():
            print(f"{key}: {np.array2string(values, precision=8, threshold=10000)}")
    return averages
