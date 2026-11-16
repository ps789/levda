"""Compare the shared core with an unchanged pre-refactor shallow-water release.

Pass --reference-dir pointing to the old directory containing
plot_resnet_data_lbfgs_ensemble.py and src/. This developer-only comparison does
not add a legacy implementation or original-tree dependency to the examples.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from levda_core.assimilation import assimilate
from levda_core.metrics import evaluate_ensemble
from levda_core.runner import average, load_data, load_model, prepare_observations, set_seed
from shallow_water.problem import PROBLEM


@contextmanager
def reference_module(folder):
    previous = {key: module for key, module in sys.modules.items()
                if key == "src" or key.startswith("src.")}
    for key in previous:
        del sys.modules[key]
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(
            "_shallow_water_reference", folder / "plot_resnet_data_lbfgs_ensemble.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(folder))
        for key in list(sys.modules):
            if key == "src" or key.startswith("src."):
                del sys.modules[key]
        sys.modules.update(previous)


def compare(actual, expected, label):
    actual, expected = torch.as_tensor(actual), torch.as_tensor(expected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=False,
                               msg=lambda message: f"{label}: {message}")
    return float(torch.max(torch.abs(actual - expected)))


def validate(args):
    device = torch.device(args.device)
    data = load_data(PROBLEM.folder / "data/sample_data.pth", args.max_steps)
    model_data = {key: data[key].to(device) for key in ("x", "u", "dt")}
    normalization = PROBLEM.load_normalization()
    truth = normalization.normalize_back(data["y"].numpy())
    results = []
    with reference_module(args.reference_dir.resolve()) as reference:
        original_data = load_data(reference.DEFAULT_DATA_PATH, args.max_steps)
        for key in data:
            compare(data[key], original_data[key], f"sample data: {key}")
        for seed in args.seeds:
            set_seed(seed)
            original = reference.build_model(device).eval()
            reference.load_model(original, reference.DEFAULT_DYN_CKPT, reference.DEFAULT_REC_CKPT, device)
            original_rng = torch.get_rng_state()
            original_observations = reference.prepare_observation(data["y"], args.obs_sigma, device)
            set_seed(seed)
            model = load_model(PROBLEM, device)
            torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
            observations = prepare_observations(data, PROBLEM, device, args.obs_sigma)
            observation_error = compare(observations, original_observations, "observations")
            reference_normalization = reference.Normalize([-0.2, -0.2, -0.5], [0.2, 0.2, 1.0])
            compare(truth, reference_normalization.normalize_back(data["y"].numpy()), "normalization")
            with torch.no_grad():
                expected_latent = original(model_data, device=device, latent_state=True)
                latent_error = compare(model(model_data, latent_state=True), expected_latent, "forecast")
            for trial in range(args.initialization_start, args.initialization_start + args.num_inits):
                trial_seed = seed + trial + PROBLEM.initialization_seed_offset
                set_seed(trial_seed)
                print(f"seed={seed}, initialization={trial + 1}: original assimilation", flush=True)
                start = time.perf_counter()
                expected, parameters = original.forward_data_assimilation_lbfgs_with_dynamics_multistep_ensemble_cov(
                    model_data, original_observations, device=device, output_ensemble=True,
                    obs_R=PROBLEM.assimilation.observation_weight,
                    alpha_reg=PROBLEM.assimilation.alpha_regularization_weight,
                )
                original_seconds = time.perf_counter() - start
                set_seed(trial_seed)
                print(f"seed={seed}, initialization={trial + 1}: shared-core assimilation "
                      f"(original {original_seconds:.2f}s)", flush=True)
                start = time.perf_counter()
                actual = assimilate(model, model_data, observations, PROBLEM.assimilation,
                                    device=device, decode_chunk_size=args.decode_chunk_size)
                clean_seconds = time.perf_counter() - start
                checks = {
                    "construction_rng_equal": True,
                    "observation_max_abs_error": observation_error,
                    "forecast_latent_max_abs_error": latent_error,
                    "analysis_latent_max_abs_error": compare(
                        torch.stack([model.state_history[i] for i in model.analysis_indices], dim=1),
                        torch.stack(original.state_history, dim=1), "analysis states"),
                    "forecast_parameter_max_abs_error": compare(
                        model.forecast_parameter_history.cpu(), parameters.detach().cpu(), "parameters"),
                    "decoded_ensemble_max_abs_error": compare(actual, expected, "full decoded ensemble"),
                }
                expected_physical = reference_normalization.normalize_back(expected.numpy())
                physical = normalization.normalize_back(actual.numpy())
                prediction = np.mean(expected_physical, axis=0, keepdims=True)
                expected_error = np.sqrt(np.sum((prediction - truth) ** 2, axis=(0, 2, 3)) /
                                         np.sum(truth ** 2, axis=(0, 2, 3)))
                expected_ser = reference.compute_ser_simple(expected_physical, truth)
                metrics = evaluate_ensemble(physical, truth)
                checks["relative_rmse_max_abs_error"] = compare(metrics["relative_rmse"], expected_error, "relative RMSE")
                checks["ser_max_abs_error"] = compare(metrics["ser"], expected_ser, "SER")
                result = {"seed": seed, "initialization_seed": trial_seed,
                          "steps": int(data["y"].shape[1]), "analysis_steps": len(model.analysis_indices),
                          "ensemble_shape": list(actual.shape), "checks": checks, "passed": True,
                          "original_seconds": original_seconds, "clean_seconds": clean_seconds,
                          "mean_relative_rmse": float(np.mean(expected_error)),
                          "mean_ser": float(np.mean(expected_ser)),
                          "relative_rmse": metrics["relative_rmse"].tolist(),
                          "ser": metrics["ser"].tolist()}
                results.append(result)
                print(json.dumps({key: value for key, value in result.items()
                                  if key not in ("relative_rmse", "ser")}, indent=2), flush=True)
                del expected, actual, expected_physical, physical
                gc.collect()
    report = {"torch": torch.__version__, "device": str(device), "results": results,
              "configuration": asdict(PROBLEM.assimilation), "obs_sigma": args.obs_sigma,
              "mean_relative_rmse": float(average([r["mean_relative_rmse"] for r in results], finite_only=True)),
              "mean_ser": float(average([r["mean_ser"] for r in results], finite_only=True)),
              "decode_chunk_size": args.decode_chunk_size, "all_passed": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"All {len(results)} comparisons passed; average relative RMSE="
          f"{report['mean_relative_rmse']:.8f}, average SER={report['mean_ser']:.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--num-inits", type=int, default=5)
    parser.add_argument("--initialization-start", type=int, default=0)
    parser.add_argument("--obs-sigma", type=float, default=PROBLEM.default_obs_sigma)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--decode-chunk-size", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/shallow_water_comparison.json")
    args = parser.parse_args()
    if args.num_inits < 1 or args.initialization_start < 0 or args.obs_sigma < 0 or args.decode_chunk_size < 0:
        parser.error("Need num-inits >= 1, initialization-start >= 0, obs-sigma >= 0, decode-chunk-size >= 0.")
    if args.max_steps is not None and args.max_steps <= PROBLEM.assimilation.smoothing_steps:
        parser.error("Need more time steps than the smoothing window.")
    torch.set_num_threads(4)
    validate(args)
