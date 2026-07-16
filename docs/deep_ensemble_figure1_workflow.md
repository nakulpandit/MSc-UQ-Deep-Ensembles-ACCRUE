# Deep Ensemble Figure 1 workflow

This phase begins the Deep Ensembles paper reproduction with the cubic toy
regression experiment in Section 3.2. It comes before the UCI benchmarks so that
the probabilistic network, adversarial training and ensemble variance decomposition
can be checked visually and numerically on a known data-generating process.

## What is reproduced

- 20 noisy observations with `y = x^3 + Normal(0, 3^2)`.
- A Gaussian network that predicts both mean and variance.
- Gaussian negative log likelihood training.
- Five independently initialized ensemble members.
- Moment-matched ensemble variance from Equation 4 of the paper.
- FGSM adversarial examples using 1% of the observed input range.
- The primary 40-epoch, Adam learning-rate `0.1` configuration.

The established project configuration samples `x` uniformly from `[-4, 4]`
and uses one 100-unit ReLU hidden layer. The paper refers to the architecture and
data protocol of Hernandez-Lobato and Adams rather than restating all of those
details, so these choices are recorded explicitly rather than presented as newly
specified by the Deep Ensembles paper.

The final workflow uses 20 independently generated training sets to measure
run-to-run variation. The paper does not specify a repeated-run count for Figure
1, so this is a robustness extension; the first run supplies the reproduction
figure.

## Outputs and metrics

The runner saves:

- `deep_ensemble_figure1_paper_config.png`;
- `figure1_predictions.csv`;
- `per_run_metrics.csv`;
- `summary.json`.

Metrics are reported overall, inside the training domain `[-4, 4]`, and outside
it. They include RMSE against noisy observations, RMSE against the true cubic
curve, NLL, one/two/three-sigma coverage, mean aleatoric/epistemic/total standard
deviation, and in-domain sigma MAE/RMSE against the true noise scale of 3.

## Branch and installation

After the neural ACCRUE W result is merged:

```bash
git switch main
git pull
git switch -c feature/deep-ensemble-figure1

python -m pip install -e '.[dev]'
pytest
```

## Debug run

```bash
uq-reproduce-deep-ensemble-figure1 --mode debug
uq-validate-deep-ensemble \
  results/development/deep_ensemble_figure1_debug \
  --mode debug --runs 1
```

The debug mode only checks that the pipeline runs. Its results are not research
results and must not be copied into the paper-reproduction directory.

## Development run

```bash
uq-reproduce-deep-ensemble-figure1 --mode development
uq-validate-deep-ensemble \
  results/development/deep_ensemble_figure1_development \
  --mode development --runs 5
```

Inspect the figure and summary before the final run:

```bash
open results/development/deep_ensemble_figure1_development/deep_ensemble_figure1_paper_config.png
cat results/development/deep_ensemble_figure1_development/summary.json
```

## Final paper-configuration run

```bash
uq-reproduce-deep-ensemble-figure1 --mode final
uq-validate-deep-ensemble \
  results/paper_reproductions/deep_ensemble_figure1 \
  --mode final --runs 20

open results/figures/deep_ensemble_figure1_paper_config.png
cat results/paper_reproductions/deep_ensemble_figure1/summary.json
```

Commit final results separately from the implementation:

```bash
git switch -c results/deep-ensemble-figure1
git add results/paper_reproductions/deep_ensemble_figure1 \
  results/figures/deep_ensemble_figure1_paper_config.png
git commit -m "Add Deep Ensemble Figure 1 reproduction results"
git push -u origin results/deep-ensemble-figure1
```

## Long-training diagnostic

The 2,000-epoch, learning-rate `0.001` configuration is deliberately separate:

```bash
uq-reproduce-deep-ensemble-figure1 --mode long_diagnostic
```

It is useful for diagnosing under-training, but it is not the primary paper
configuration and must not replace the 40-epoch result.

After Figure 1 is validated, the next Deep Ensembles phase reproduces Table 1 on
the original regression benchmark datasets using RMSE and NLL.
