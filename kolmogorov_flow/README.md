# Kolmogorov Flow

From the repository root:

```bash
python -m kolmogorov_flow.run --device cuda:0 --seed 42
```

Or run `python run.py` in this folder. This uses the full 40-step trajectory 0,
20 ensemble members, 10 L-BFGS iterations, learning rate 0.1, and 10% multiplicative
observation noise, matching `plot_resnet_data_lbfgs_kolmogorov_ensemble.py` in
the original experiment. The shared code is in the sibling `levda_core` package.

Included are the unchanged `dyn_1999.ckpt`, `retrained_rec_1999.ckpt`, the exact
original Gaussian normalization, and one complete test trajectory. See
`assets_manifest.json` for provenance and the root README for metric definitions,
the legacy-compatibility choices, and original-versus-clean validation.

The runner prints physical-space ensemble-mean relative RMSE and SER and saves
nothing. Add `--num-inits 5` for initialization averages or `--print-curves`
for the time series.
