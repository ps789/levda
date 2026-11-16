"""One-time asset extraction from the original, trusted local experiment files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = Path("/work/pengpeng/data-assimilation")
ORIGINAL = Path("/work2/phillip/latent-assimilation")
SOURCES = {
    "kolmogorov_flow": {
        "data": EXTERNAL / "kolmogorov_flow/data/data_observation_500_1500_150x150_seed_0_resnet_fourier_10_gamma_0.5_resd_14.pth",
        "checkpoints": EXTERNAL / "kolmogorov_flow/saved_model/cplx_Re500_1500_150x150_resnet_fourier_10_gamma_0.5_resd_14",
        "stats": EXTERNAL / "kolmogorov_flow/saved_model/cplx_Re500_1500_150x150/mean_std.npz",
        "files": ["dyn_1999.ckpt", "retrained_rec_1999.ckpt"],
        "script": ORIGINAL / "kolmogorov_flow/plot_resnet_data_lbfgs_kolmogorov_ensemble.py",
    },
    "planetswe": {
        "data": EXTERNAL / "planetswe/saved_model/parameter_shuf_t21d/observation_32.pth",
        "checkpoints": EXTERNAL / "planetswe/saved_model/parameter_shuf_t21d",
        "stats": EXTERNAL / "planetswe/saved_model/parameter_shuf_t21d/mean_std.npz",
        "files": ["dyn_1799.ckpt", "rec_1799.ckpt"],
        "script": ORIGINAL / "planetswe/test_batch_average_ensemble_voda.py",
    },
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(name, index):
    source = SOURCES[name]
    folder = ROOT / name
    (folder / "data").mkdir(parents=True, exist_ok=True)
    (folder / "saved_model").mkdir(exist_ok=True)
    data = torch.load(source["data"], map_location="cpu", weights_only=True, mmap=True)["data_test"]
    if not 0 <= index < data["y"].shape[0]:
        raise ValueError("Trajectory index is outside the test set.")
    sample = {key: data[key][index:index + 1].contiguous().clone() for key in ("x", "u", "y")}
    sample["dt"] = data["dt"].clone()
    for key in sample:
        original = data[key] if key == "dt" else data[key][index:index + 1]
        assert torch.equal(sample[key], original), key
    path = folder / "data/sample_data.pth"
    torch.save({"data_test": sample}, path)
    copies = {}
    for filename in source["files"]:
        origin = source["checkpoints"] / filename
        dest = folder / "saved_model" / filename
        shutil.copy2(origin, dest)
        assert digest(origin) == digest(dest)
        copies[filename] = {"source": str(origin), "sha256": digest(dest)}
    dest = folder / "saved_model/mean_std.npz"
    shutil.copy2(source["stats"], dest)
    assert digest(source["stats"]) == digest(dest)
    stats = np.load(dest, allow_pickle=False)
    report = {"original_entrypoint": str(source["script"]), "original_data": str(source["data"]),
              "trajectory_index": index, "all_sample_tensors_equal_original": True,
              "sample_sha256": digest(path), "sample_bytes": path.stat().st_size,
              "shapes": {key: list(value.shape) for key, value in sample.items()},
              "checkpoints": copies, "normalization_source": str(source["stats"]),
              "normalization_sha256": digest(dest),
              "normalization": {key: stats[key].tolist() for key in stats.files},
              "note": "One full-length test trajectory; no training/validation data or auxiliary encoders included."}
    (folder / "assets_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(name, json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=[*SOURCES, "all"], default="all")
    parser.add_argument("--trajectory-index", type=int, default=0)
    args = parser.parse_args()
    for problem in SOURCES if args.problem == "all" else [args.problem]:
        prepare(problem, args.trajectory_index)
