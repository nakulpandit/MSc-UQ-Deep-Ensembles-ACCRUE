# MSc UQ: Deep Ensembles and ACCRUE

Comparing two approaches to uncertainty quantification in neural-network regression

1. Deep Ensembles
2. ACCRUE
3. Direct comparison using identical data, splits, seeds, and metrics

## Repository structure

- `deep_ensembles.ipynb` — implementation and evaluation of Deep Ensembles
- `accrue.ipynb` — implementation and evaluation of ACCRUE
- `comparison.ipynb` — controlled comparison of both methods
- `requirements.txt` — Python dependencies

## Research objective

The objective is to compare the predictive accuracy, uncertainty quality,
calibration, and out-of-distribution behaviour of Deep Ensembles and ACCRUE
under a reproducible experimental setup.

## Initial experimental plan

Both methods will use:

- the same synthetic regression dataset;
- the same train, validation, and test splits;
- fixed random seeds;
- comparable neural-network architectures;
- identical evaluation points and plots.

Initial metrics will include:

- RMSE;
- Gaussian negative log-likelihood;
- empirical interval coverage;
- mean prediction-interval width;
- calibration behaviour;
- uncertainty under distribution shift.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
