# Levda Shallow Water Review Release

This folder is a minimal, self-contained shallow-water release for the LBFGS ensemble baseline.

Included:
- a trimmed `plot_resnet_data_lbfgs_ensemble.py` entrypoint
- only the shallow-water model code needed by that script
- a one-trajectory sample dataset extracted from the original test set
- the two checkpoints used by the baseline (`dyn_1999.ckpt` and `retrained_rec_1999.ckpt`)

Run:

```bash
python plot_resnet_data_lbfgs_ensemble.py
```

The script prints the average relative RMSE and average SER over multiple random initializations and does not save any files.

