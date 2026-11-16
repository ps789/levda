"""Run unchanged original model methods against the clean port with identical RNG states.

Only this optional developer regression script requires the original tree and DeepXDE.
The released examples do not import either one.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import warnings

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch

from levda_core.assimilation import assimilate
from levda_core.metrics import evaluate_ensemble
from levda_core.problems import PROBLEMS
from levda_core.runner import load_data, load_model, prepare_observations, set_seed


@contextmanager
def original_modules(folder):
    previous = {key: value for key, value in sys.modules.items() if key == "src" or key.startswith("src.")}
    for key in previous:
        del sys.modules[key]
    sys.path.insert(0, str(folder))
    try:
        yield importlib.import_module("src.fourier_ldnet")
    finally:
        sys.path.remove(str(folder))
        for key in list(sys.modules):
            if key == "src" or key.startswith("src."):
                del sys.modules[key]
        sys.modules.update(previous)


def compare(actual, expected, label, rtol=1e-5, atol=2e-6):
    actual = torch.as_tensor(actual)
    expected = torch.as_tensor(expected)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=False,
                               msg=lambda message: f"{label}: {message}")
    return float(torch.max(torch.abs(actual - expected)))


def original_observations(data, name, device):
    if name == "kolmogorov_flow":
        observation = data["y"]
        observation = data["y"] + torch.randn_like(observation) * 0.1 * data["y"]
        return observation.reshape(1, observation.shape[1], 150, 150, 2)[:, :, ::15, ::15].to(device)
    observation = data["y"].to(device).reshape(1, data["y"].shape[1], 512, 256, 3)[:, :, ::32, ::32]
    return observation + 0.1 * torch.randn_like(observation) * observation


def validate(name, seed, args):
    problem = PROBLEMS[name]
    manifest = json.loads((problem.folder / "assets_manifest.json").read_text())
    data = load_data(problem.folder / "data/sample_data.pth", args.max_steps)
    source_data = torch.load(manifest["original_data"], map_location="cpu", weights_only=True, mmap=True)["data_test"]
    index = manifest["trajectory_index"]
    for key in ("x", "u", "y", "dt"):
        original = source_data[key] if key == "dt" else source_data[key][index:index + 1]
        if args.max_steps and key in ("x", "y"):
            original = original[:, :args.max_steps]
        torch.testing.assert_close(data[key], original, rtol=0, atol=0)
    del source_data
    device = torch.device(args.device)
    module_folder = args.original_root / "kolmogorov_flow" if name == "kolmogorov_flow" else args.original_root
    with original_modules(module_folder) as module:
        original_model = module.ResFourierLDNN(
            problem.fourier_size, list(problem.dynamics_sizes), list(problem.decoder_sizes),
            activation="relu", kernel_initializer="Glorot normal",
        ).to(device).eval()
    for component, filename in [("dyn", problem.dynamics_checkpoint), ("rec", problem.decoder_checkpoint)]:
        path = Path(manifest["checkpoints"][filename]["source"])
        getattr(original_model, component).load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model = load_model(problem, device)
    model_data = {key: data[key].to(device) for key in ("x", "u", "dt")}
    checks = {}
    with torch.no_grad():
        original_latent = original_model(model_data, device=device, latent_state=True)
        clean_latent = model(model_data, device=device, latent_state=True)
        checks["unassimilated_latent_max_abs_error"] = compare(clean_latent, original_latent, "latent forecast", 0, 0)
        coordinates = model_data["x"][:, -1, ::100]
        state = original_latent[:, -1:].expand(1, coordinates.shape[1], -1)
        checks["decoder_probe_max_abs_error"] = compare(
            model.rec(coordinates, state), original_model.rec(coordinates, state), "decoder probe", 0, 0
        )
    set_seed(seed)
    observations = original_observations(data, name, device)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    set_seed(seed)
    clean_observations = prepare_observations(data, problem, device)
    checks["observation_max_abs_error"] = compare(clean_observations, observations, "observation preparation", 0, 0)
    del clean_observations
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    print(f"{name} seed={seed}: original {data['y'].shape[1]}-step assimilation starting", flush=True)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="levda_original_") as tmp:
        previous_cwd = Path.cwd()
        os.chdir(tmp)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Using a target size")
                reference = original_model.forward_data_assimilation_lbfgs_with_dynamics_multistep_ensemble_cov(
                    model_data, observations, device=device, output_ensemble=True
                )
        finally:
            os.chdir(previous_cwd)
    reference_seconds = time.perf_counter() - start
    reference_states = torch.stack(original_model.state_history, dim=1).cpu()
    del original_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    print(f"{name} seed={seed}: clean assimilation starting (original {reference_seconds:.2f}s)", flush=True)
    start = time.perf_counter()
    clean = assimilate(model, model_data, observations, problem.assimilation, device=device,
                       decode_chunk_size=args.decode_chunk_size)
    clean_seconds = time.perf_counter() - start
    checks["assimilation_latent_max_abs_error"] = compare(
        torch.stack(model.state_history, dim=1).cpu(), reference_states, "assimilated latent trajectory"
    )
    checks["decoded_ensemble_max_abs_error"] = compare(clean, reference, "decoded full ensemble")
    stats = np.load(problem.folder / "saved_model/mean_std.npz")
    truth = data["y"].numpy() * stats["std"] + stats["mean"]
    reference_physical = reference.numpy() * stats["std"] + stats["mean"]
    clean_physical = clean.numpy() * stats["std"] + stats["mean"]
    reference_metrics = evaluate_ensemble(reference_physical, truth)
    clean_metrics = evaluate_ensemble(clean_physical, truth)
    for metric in reference_metrics:
        checks[f"{metric}_max_abs_error"] = compare(clean_metrics[metric], reference_metrics[metric], metric)
    # Independently evaluate the exact original entrypoint's metric formula.
    if name == "kolmogorov_flow":
        mean = np.mean(reference_physical, axis=0, keepdims=True)
        original_error = np.sqrt(np.sum((mean - truth) ** 2, axis=(0, 2, 3)) /
                                 np.sum(truth ** 2, axis=(0, 2, 3)))
        original_metric = "relative_rmse"
    else:
        original_error = np.sqrt(np.mean((reference_physical - truth) ** 2, axis=(0, 2, 3)) /
                                 np.mean(truth ** 2, axis=(0, 2, 3)))
        original_metric = "member_rms_relative_rmse"
    checks["original_script_metric_max_abs_error"] = compare(clean_metrics[original_metric], original_error, "original-script metric")
    result = {"problem": name, "seed": seed, "steps": int(data["y"].shape[1]),
              "ensemble_shape": list(clean.shape), "configuration": asdict(problem.assimilation),
              "checks": checks, "passed": True,
              "original_seconds": reference_seconds, "clean_seconds": clean_seconds,
              "clean_decode_chunk_size": args.decode_chunk_size,
              "metrics": {key: {"original_mean": float(np.mean(reference_metrics[key])),
                                 "clean_mean": float(np.mean(value)),
                                 "original_final": float(reference_metrics[key][-1]),
                                 "clean_final": float(value[-1]),
                                 "original_curve": reference_metrics[key].tolist(),
                                 "clean_curve": value.tolist()}
                          for key, value in clean_metrics.items()},
              "original_script_metric": original_metric,
              "data_verified_against_original": True}
    print(json.dumps({key: value for key, value in result.items() if key != "metrics"}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, default=Path("/work2/phillip/latent-assimilation"))
    parser.add_argument("--problem", choices=[*PROBLEMS, "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--decode-chunk-size", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/original_comparison.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = []
    for problem in PROBLEMS if args.problem == "all" else [args.problem]:
        for seed in args.seeds:
            results.append(validate(problem, seed, args))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"torch": torch.__version__, "device": args.device,
                                              "results": results}, indent=2) + "\n")
