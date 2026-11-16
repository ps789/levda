# Tsunami / Shallow Water

From the repository root:

```bash
python -m shallow_water.run --device cuda:0 --seed 42
```

Or run `python run.py` in this folder. The example uses the same `levda_core`
model, L-BFGS smoother, metrics, and runner as Kolmogorov flow and PlanetSWE.
Architecture and experiment settings live in `problem.py`, not in the core.

Defaults match the pre-refactor shallow-water release: one complete 51-step
trajectory, 30 ensemble members, five initializations, 5% multiplicative sensor
noise, learning rate 0.1, up to 40 L-BFGS iterations with strong-Wolfe line search,
and alpha-increment row-Gram regularization of weight 1. The final five steps
are forecast-only because the original smoother requires a complete look-ahead
window. Observations are fixed across initializations, using seeds 43 through
47 for seed 42. Model construction consumes the original random draws before
observation noise is generated.

The original dataset and both checkpoint files are unchanged, now in `data/`
and `saved_model/`. The original min-max physical-space normalization is retained.
The runner prints relative RMSE of the ensemble mean and SER, without saving
output files or rerunning divergent trials. As before, the final averages
exclude nonfinite initialization means and explicitly report the finite counts.

For a quick check that includes an analysis update:

```bash
python -m shallow_water.run --device cpu --max-steps 3 --smoothing-steps 1 --ensemble-size 3 --max-iter 1 --num-inits 1
```

The old command `python shallow_water/shallow_water/plot_resnet_data_lbfgs_ensemble.py`
still works through a thin wrapper. Its `--obs-r`, `--alpha-reg`, `--dyn-ckpt`,
and `--rec-ckpt` arguments are supported by the shared runner.

