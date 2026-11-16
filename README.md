# LEVDA Review Examples

Minimal, self-contained checkpoint-based inference and assimilation examples for
Kolmogorov flow, PlanetSWE, and tsunami/shallow water. All three use the shared
`levda_core` framework. Each problem folder owns its `problem.py` configuration,
observation preparation, `run.py` entrypoint, data, and checkpoints; the core
contains no problem registry or problem-specific presets.

## Setup and Run

Use Python 3.10 or newer with the dependencies in `requirements.txt`.

```bash
cd /work2/phillip/levda
python -m pip install -r requirements.txt
python -m kolmogorov_flow.run --device cuda:0 --seed 42
python -m planetswe.run --device cuda:0 --seed 42
python -m shallow_water.run --device cuda:0 --seed 42
```

Paths are relative to the code, not the working directory. Direct invocation also
works, for example `python /work2/phillip/levda/planetswe/run.py`.
`--help` lists the options. CUDA is selected automatically if available;
`--device cpu` runs on CPU. No output files are written by the example runners.

The installed environment
`/work2/phillip/anaconda3/envs/swe_era5/bin/python` has the release dependencies
and was used for the standalone smoke tests. Original-tree comparisons additionally
use `requirements-validation.txt`; those dependencies are not needed to run examples.

To average independently initialized ensembles while keeping the noisy
observations fixed:

```bash
python -m kolmogorov_flow.run --num-inits 5 --seed 42
python -m planetswe.run --num-inits 5 --seed 42
python -m shallow_water.run --num-inits 5 --seed 42
```

For a short installation check rather than a full benchmark:

```bash
python -m planetswe.run --device cpu --max-steps 3 --ensemble-size 3 --max-iter 1
```

Shallow water defaults to five initializations and requires a full smoothing
window for each analysis, unlike the Fourier examples. For a short check with
an actual analysis update, use:

```bash
python -m shallow_water.run --device cpu --max-steps 3 --smoothing-steps 1 --ensemble-size 3 --max-iter 1 --num-inits 1
```

The old shallow-water script path remains a thin command-line wrapper. Its
duplicate `src` model code has been removed; see `shallow_water/README.md` for
the preserved experiment settings. Installing with `python -m pip install .`
includes all three problems and their assets. After cloning, use Git LFS to
download the PlanetSWE sample (`git lfs pull`).

## Validation

Fast CPU tests and relocated, isolated-Python checks:

```bash
python -m unittest discover -s tests -v
python scripts/check_portability.py
```

`scripts/validate_against_original.py` compares the Fourier examples against the
original experiment tree and checkpoints. `scripts/validate_shallow_water.py`
compares against an unchanged pre-refactor shallow-water release. A reference
copy can be extracted from the repository history without changing the worktree:

```bash
reference=$(mktemp -d)
git archive 5381cf8 shallow_water/shallow_water | tar -x -C "$reference"
python scripts/validate_shallow_water.py --reference-dir "$reference/shallow_water/shallow_water" --device cuda:0
```

The comparison checks the full 51-step trajectory over five initializations,
including exact observation RNG, forecast and analysis states, parameters,
decoded ensemble, relative RMSE, and SER. Example runners still save no output;
only developer validation scripts write reports under `validation/`.
