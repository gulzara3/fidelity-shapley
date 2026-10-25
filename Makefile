.PHONY: install test lint evidence smoke reproduce figures verify clean

PY ?= python
EXP ?= outputs/repro

install:            ## install the package with dev tools
	$(PY) -m pip install -e ".[dev]"

test:               ## unit tests + evidence checks (no GPU, no download)
	$(PY) -m pytest

evidence:           ## re-derive every headline number from results/
	$(PY) -m pytest tests/test_evidence.py -v

lint:
	ruff check src tests scripts

smoke:              ## fast end-to-end CPU check of the installation (a few minutes)
	fidelity-shapley run --config configs/smoke.yaml

reproduce:          ## recompute every stage from the archived checkpoints
	bash scripts/reproduce_from_checkpoints.sh $(EXP)

full:               ## full paper run from scratch (GPU recommended, ~2–2.5 h on an L4)
	fidelity-shapley run --config configs/paper.yaml

verify:             ## compare $(EXP) against results/
	fidelity-shapley verify --candidate $(EXP) --reference results --report $(EXP)/reproduction_report.md

figures:            ## regenerate the publication figures from results/
	fidelity-shapley figures --results results --out results/figures/paper

clean:
	rm -rf outputs .pytest_cache .ruff_cache build dist src/*.egg-info
