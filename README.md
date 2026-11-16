# LEVDA Review Examples

Minimal, self-contained checkpoint-based inference and assimilation examples for
Kolmogorov flow, PlanetSWE, and the existing tsunami/shallow-water release.

## Setup and Run

Use Python 3.10 or newer with the dependencies in `requirements.txt`.

```bash
cd /work2/phillip/levda
python -m pip install -r requirements.txt
python -m kolmogorov_flow.run --device cuda:0 --seed 42
python -m planetswe.run --device cuda:0 --seed 42
```

Paths are relative to the code, not the working directory. Direct invocation also
works, for example `python /work2/phillip/levda/planetswe/run.py`.
`--help` lists the options. CUDA is selected automatically if available;
`--device cpu` runs on CPU. No output files are written by the example runners.

The installed environment
`/work2/phillip/anaconda3/envs/swe_era5/bin/python` has the release dependencies
and was used for the standalone smoke tests. The separate validation environment
is described below; it is not required for ordinary use.

To average independently initialized ensembles while keeping the noisy
observations fixed:

```bash
python -m kolmogorov_flow.run --num-inits 5 --seed 42
python -m planetswe.run --num-inits 5 --seed 42
```

For a short installation check rather than a full benchmark:

```bash
python -m planetswe.run --device cpu --max-steps 3 --ensemble-size 3 --max-iter 1
```

Changing these settings changes the experiment; use the defaults to reproduce
the original entry points. `--decode-chunk-size 4096` limits physical-grid
reconstruction memory. It does not split or change the assimilation loss.
Use `--decode-chunk-size 0` for the original monolithic full-grid decode.
