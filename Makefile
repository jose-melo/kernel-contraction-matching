PYTHON ?= python

.PHONY: help datasets test kcm collapse discriminators backbone aggregate clean

help:
	@echo "datasets       download the 47 ADBench .npz files"
	@echo "test           run the test suite, including the docstring examples"
	@echo "kcm            Sec. 5 KCM benchmark over 47 datasets"
	@echo "collapse       Sec. 3 peak-to-final gap, 6 detectors"
	@echo "discriminators Sec. 3 paradox-vs-overfitting grid"
	@echo "backbone       Sec. 6 backbone independence"
	@echo "aggregate      per-section summary CSVs"
	@echo ""
	@echo "pip install -e .            KCM only, no torch"
	@echo "pip install -e '.[paper]'   + torch and pandas, needed by kcm / collapse"

datasets:
	$(PYTHON) scripts/download_datasets.py


test:
	$(PYTHON) -m pytest -q

kcm:
	$(PYTHON) -m karkcm.experiments.kcm_benchmark

collapse:
	$(PYTHON) -m karkcm.experiments.collapse_gap

discriminators:
	$(PYTHON) -m karkcm.experiments.discriminators

backbone:
	$(PYTHON) -m karkcm.experiments.backbone

aggregate:
	$(PYTHON) -m karkcm.experiments.aggregate

clean:
	find . -name __pycache__ -type d -exec rm -rf {} +
	rm -rf build dist *.egg-info
