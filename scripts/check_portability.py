"""Exercise all three problem folders independently of the original workspace."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent


def main():
    reports = []
    with tempfile.TemporaryDirectory(prefix="levda_portability_") as temporary:
        destination = Path(temporary) / "review_copy"
        for name in ("levda_core", "kolmogorov_flow", "planetswe", "shallow_water"):
            shutil.copytree(ROOT / name, destination / name,
                            ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("kolmogorov_flow", "planetswe", "shallow_water"):
            command = [sys.executable, "-I", str(destination / name / "run.py"),
                       "--device", "cpu", "--max-steps", "2", "--ensemble-size", "3",
                       "--max-iter", "1", "--threads", "2", "--num-inits", "2"]
            if name == "shallow_water":
                command.extend(["--smoothing-steps", "1"])
            result = subprocess.run(command, cwd=temporary, text=True, capture_output=True,
                                    check=True, timeout=120)
            assert "Average relative RMSE over 2 initialization(s)" in result.stdout
            assert not (destination / "latent-assimilation").exists()
            assert "nan" not in result.stdout.lower()
            reports.append({"problem": name, "isolated_python": True, "device": "cpu",
                            "steps": 2, "members": 3, "max_iter": 1, "initializations": 2,
                            "smoothing_steps": 1 if name == "shallow_water" else 5,
                            "ran_from_unrelated_working_directory": True,
                            "original_tree_not_copied": True, "stdout": result.stdout,
                            "passed": True})
            print(name, "relocated isolated CPU run passed", flush=True)
    output = ROOT / "validation/portability.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({"python": sys.executable, "results": reports}, indent=2) + "\n")


if __name__ == "__main__":
    main()
