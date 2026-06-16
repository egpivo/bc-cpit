# Environment Setup

## Quick start (conda)

```bash
make conda-env
conda activate cpit
make test
```

- `envs/conda/environment_cpit.yml` — Conda env (Python ≥3.11, numpy, scipy, scikit-learn, pandas, matplotlib, pygam)

## Quick start (venv)

```bash
make venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
make install
```

## Update conda env after changing `environment_cpit.yml`

```bash
make conda-update
```

## Jupyter

```bash
make jupyter   # registers kernel and launches JupyterLab
```
