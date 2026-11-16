# PlanetSWE

From the repository root:

```bash
python -m planetswe.run --device cuda:0 --seed 42
```

Or run `python run.py` in this folder. This uses the full 63-step trajectory 0,
20 ensemble members, 10 L-BFGS iterations, learning rate 0.001, and 10%
multiplicative observation noise, matching `test_batch_average_ensemble_voda.py`
and its shared `src/fourier_ldnet.py` in the original experiment.

Included are the unchanged `dyn_1799.ckpt`, `rec_1799.ckpt`, the exact original
Gaussian normalization, and one complete test trajectory. See
`assets_manifest.json` for provenance. Only NumPy and PyTorch are needed;
there is no atmosphere simulator or training dependency.

The runner prints physical-space ensemble-mean relative RMSE and SER. It also
prints the original script's different member-RMS error for direct comparison.
No output files are saved. Add `--num-inits 5` for initialization averages,
`--print-curves` for the full time series, or `--max-steps 3` for a short check.
Full-grid output decoding is chunked to reduce GPU memory, not the assimilation
loss. Original-versus-clean regression reports and details are in the root
README and `validation` folder.
