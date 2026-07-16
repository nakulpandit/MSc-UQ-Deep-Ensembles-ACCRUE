# MSc UQ: Deep Ensembles vs ACCRUE

Clean, paper-led implementation for Nakul Pandit's MSc research project on
uncertainty quantification in neural-network regression.

The project compares two Gaussian predictive-uncertainty strategies:

1. **Deep Ensembles** (Lakshminarayanan, Pritzel & Blundell, 2017): each independently
   initialized network jointly predicts the conditional mean and variance by minimizing
   Gaussian negative log-likelihood (NLL). Predictions are combined by moment matching.
2. **ACCRUE** (Camporeale & Carè, 2021): a deterministic mean predictor is trained first;
   a second model then estimates input-dependent standard deviation by balancing Gaussian
   continuous ranked probability score (CRPS) and an analytic reliability score (RS).

## Experimental stages

| Stage | Purpose | Data | Primary outputs |
|---|---|---|---|
| 1 | Unit-check equations and pipeline | ACCRUE G, Y, W and 5D synthetic data | recovered noise, CRPS, RS, calibration |
| 2 | Reproduce paper regression tables | Original UCI benchmark datasets | RMSE/NLL for Deep Ensembles; CRPS/calibration error for ACCRUE |
| 3 | Fair head-to-head comparison | Shared original benchmarks | RMSE, NLL, CRPS, calibration error, coverage, sharpness, runtime |
| 4 | Generalisation study | Three shared scientific datasets, frozen before running | the same metrics and paired repeated-run statistics |

Stages are deliberately separate. Concrete, Energy Efficiency and Power Plant are present in
both papers, so they are shared **paper benchmarks**, not novel datasets.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
uq-reproduce-accrue --dataset g --runs 100
uq-validate-results results/paper_reproductions/accrue_polynomial_g --dataset g --runs 100
uq-reproduce-accrue-neural --dataset g --mode debug
uq-validate-results results/development/accrue_neural_g_debug \
  --kind neural --dataset g --mode debug --runs 1
uq-refresh-neural-results results/paper_reproductions/accrue_neural_5d
uq-reproduce-deep-ensemble-figure1 --mode debug
uq-validate-deep-ensemble results/development/deep_ensemble_figure1_debug \
  --mode debug --runs 1
uq-reproduce-toys --dataset g --method both --runs 1 --quick
```

The dedicated `uq-reproduce-accrue` command is the paper-style ACCRUE toy
reproduction. It saves per-run metrics, a summary, a Figure 4-style uncertainty
recovery plot and a reliability diagram under
`results/paper_reproductions/accrue_polynomial_<dataset>/`.

Follow [`docs/step_by_step_workflow.md`](docs/step_by_step_workflow.md) to run,
inspect, commit and push the G, Y and W reproductions one at a time.

The next paper phase uses the 50/10 neural ACCRUE model on G, Y, W and 5D.
Use debug and development modes before the 100-run paper configuration; follow
[`docs/neural_accrue_workflow.md`](docs/neural_accrue_workflow.md).

The Deep Ensembles paper phase begins with the Section 3.2 cubic regression
experiment. The debug, development, final and separately labelled long-training
workflows are documented in
[`docs/deep_ensemble_figure1_workflow.md`](docs/deep_ensemble_figure1_workflow.md).

## Reproducibility rules

- Fix and record the data split and model seed separately.
- Fit preprocessing on the training partition only.
- Give both methods identical train/validation/test rows in comparison experiments.
- Use all training rows for every Deep Ensemble member; diversity comes from initialization
  and minibatch order, matching the paper.
- Use paper-style in-sample residuals only for ACCRUE reproduction. Use five-fold
  out-of-fold residuals for the later fair comparison to prevent optimistic variance fitting.
- Never select a model or calibration parameter on the test set.
- Store per-run results before computing aggregate tables.

The detailed protocol and paper-to-code decisions are in
[`docs/research_protocol.md`](docs/research_protocol.md).
For a file-by-file explanation, see
[`docs/repository_guide.md`](docs/repository_guide.md).

## References

- Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). *Simple and Scalable
  Predictive Uncertainty Estimation using Deep Ensembles*. NeurIPS 30.
- Camporeale, E. & Carè, A. (2021). *ACCRUE: Accurate and Reliable Uncertainty
  Estimate*. International Journal for Uncertainty Quantification, 11(4), 81–94.
