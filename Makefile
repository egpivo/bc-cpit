# cpit — bias-corrected conformal PIT calibration
# Run from the cpit/ package root.

PYTHON ?= python3
PIP    ?= pip
PYTEST ?= pytest

# ── Install ──────────────────────────────────────────────────────────────────

# Editable install (uses pyproject.toml)
install:
	$(PIP) install -e ".[dev]"

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	PYTHONPATH=. $(PYTEST) tests/ -v --tb=short

test-cov:
	PYTHONPATH=. $(PYTEST) tests/ --cov=cpit --cov-report=term-missing

# ── Simulations (draft §5) ───────────────────────────────────────────────────

# Run all three designs (sequential)
run-sim: run-design1 run-design2 run-design3

# Design 1: global location-scale distortion
run-design1:
	PYTHONPATH=. $(PYTHON) -m examples.simulation.run_design1 \
		--n_reps 100 --base_seed 42 --mode all \
		--summary_path results/simulation/design1/design1_rep_summary.csv

run-design1-gam:
	PYTHONPATH=. $(PYTHON) -m examples.simulation.run_design1 \
		--n_reps 100 --base_seed 42 --mode gam \
		--summary_path results/simulation/design1/design1_rep_summary_gam.csv

# Design 2: x-dependent location-scale distortion
run-design2:
	PYTHONPATH=. $(PYTHON) -m examples.simulation.run_design2 \
		--n_reps 100 --base_seed 42 --mode all \
		--summary_path results/simulation/design2/design2_rep_summary.csv

run-design2-gam:
	PYTHONPATH=. $(PYTHON) -m examples.simulation.run_design2 \
		--n_reps 100 --base_seed 42 --mode gam \
		--summary_path results/simulation/design2/design2_rep_summary_gam.csv

# Design 3: shape misspecification (upper-tail focus)
run-design3:
	PYTHONPATH=. $(PYTHON) -m examples.simulation.run_design3 \
		--n_reps 100 --base_seed 42 --mode all \
		--summary_path results/simulation/design3/design3_rep_summary.csv

run-design3-gam:
	PYTHONPATH=. $(PYTHON) -m examples.simulation.run_design3 \
		--n_reps 100 --base_seed 42 --mode gam \
		--summary_path results/simulation/design3/design3_rep_summary_gam.csv

# PIT histogram figures for all designs
pit-figures:
	PYTHONPATH=. $(PYTHON) -m examples.simulation.plot_pit \
		--designs 1 2 3 --output-dir docs/figures --seed 42

# ── Conda environment ─────────────────────────────────────────────────────────

conda-env:
	conda env create -f envs/conda/environment_cpit.yml
	conda run -n cpit pip install -e ".[dev]"
	@echo "Run: conda activate cpit"

conda-update:
	conda env update -f envs/conda/environment_cpit.yml --prune

# ── Virtual environment (alternative) ────────────────────────────────────────

venv:
	$(PYTHON) -m venv .venv
	@echo "Then: .venv/bin/pip install -e .[dev]"

# ── Jupyter ───────────────────────────────────────────────────────────────────

jupyter:
	$(PYTHON) -m ipykernel install --user --name=cpit --display-name="Python (cpit)"
	$(PYTHON) -m jupyter lab 2>/dev/null \
		|| $(PYTHON) -m jupyter notebook 2>/dev/null \
		|| $(PYTHON) -m jupyter server 2>/dev/null \
		|| (echo "No Jupyter app found. Try: pip install jupyterlab"; exit 1)

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	rm -rf build/ dist/ *.egg-info .eggs
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name __pycache__ | xargs rm -rf 2>/dev/null || true
	find . -type d -name .pytest_cache | xargs rm -rf 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints | xargs rm -rf 2>/dev/null || true

.PHONY: install test test-cov \
        run-sim \
        run-design1 run-design1-gam \
        run-design2 run-design2-gam \
        run-design3 run-design3-gam \
        pit-figures \
        conda-env conda-update venv jupyter clean
