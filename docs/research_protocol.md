# Research protocol

## Research question

When the predictive distribution is assumed Gaussian, how does joint mean/variance learning
with a five-member Deep Ensemble compare with a two-stage ACCRUE estimator in predictive
accuracy, probabilistic quality, calibration, sharpness, uncertainty recovery, and computation?

## What each paper actually optimises

| Item | Deep Ensembles | ACCRUE |
|---|---|---|
| Mean model | Learned jointly | Trained first as a deterministic oracle |
| Uncertainty model | One variance head per ensemble member | Separate model for input-dependent standard deviation |
| Training score | Gaussian NLL | `beta * mean(CRPS) + (1-beta) * RS` |
| Diversity | Independent initialization and shuffled minibatches | Five initializations; retain lowest validation cost in paper reproduction |
| Regression paper metrics | RMSE and NLL; interval calibration example | CRPS and maximum reliability-diagram deviation |
| Key uncertainty | Aleatoric plus between-member epistemic spread | Conditional residual scale around one fixed mean model |

These methods do not estimate exactly the same uncertainty object. That distinction must be stated
when interpreting the final comparison.

## Stage 1: synthetic reproduction

ACCRUE defines four heteroscedastic generators:

- G: 100 observations, one input.
- Y: 100 observations, one input.
- W: 100 observations, one input.
- 5D: 10,000 observations, five inputs.

Targets are sampled as `Normal(f(x), sigma(x)^2)`. The paper uses 33/33/34 percent
train/validation/test splits, 100 independent runs, and a variance network with hidden widths
50 and 10. It reports a ReLU first activation, a symmetric saturating-linear second activation,
`log(sigma)` output, quasi-Newton optimization, early stopping after ten validation non-improvements,
and selection of the best of five initializations.

For the ACCRUE reproduction, the mean oracle follows the paper: a homoskedastic Gaussian process
for G/Y/W and the exact zero mean for 5D. A neural deterministic oracle will be an explicitly
labelled ablation, not silently substituted.

The Deep Ensembles paper's Figure 1 toy equation is not specified in the paper text. We will not
invent an exact data-generating equation and call it a faithful reproduction. Instead, its qualitative
toy demonstration will be reproduced on the fully specified G/Y/W generators and labelled as an
adaptation.

## Stage 2: original real-world benchmarks

Deep Ensembles uses Boston Housing, Concrete, Energy, Kin8nm, Naval Propulsion, Power Plant,
Protein, Wine, Yacht and Year Prediction MSD. It uses 20 folds, except Protein (5) and Year (one),
one ReLU hidden layer with 50 units (100 for Protein and Year), 40 epochs, and five members.

ACCRUE uses Boston Housing, Concrete, Energy, Kin8nm, Power Plant, Protein, Wine and Yacht,
with 70% training data and 50 runs. Its table reports median CRPS and calibration error, with
one standard deviation. Details not fixed by the paper (exact split seeds and some mean-network
hyperparameters) will be recorded as implementation decisions and tested for sensitivity.

Boston Housing has ethical/documentation concerns and is no longer distributed by modern
scikit-learn. It remains in the historical table for paper fidelity, but acquisition and use must be
documented; it is not required for the main new-data conclusion.

## Stage 3: fair shared comparison

Begin with Concrete, Energy and Power Plant as a low/medium-scale pipeline check because both
papers use them. For every repeated split:

1. Create one train/validation/test split and save row indices.
2. Fit feature and target scalers on training data only.
3. Train a five-member Deep Ensemble.
4. Train the deterministic ACCRUE mean model with the same capacity as one ensemble member.
5. Generate five-fold out-of-fold training residuals for ACCRUE's variance model.
6. Select hyperparameters using validation data only.
7. Evaluate once on the untouched test data.
8. Save per-example predictions and per-run metrics for paired analysis.

## Metrics

No single metric is enough. The common scorecard is:

- **Point accuracy:** RMSE and MAE.
- **Probabilistic accuracy:** Gaussian NLL and Gaussian CRPS.
- **Calibration:** maximum absolute PIT/reliability deviation; empirical coverage at 50%, 80%,
  90% and 95%; mean absolute coverage error.
- **Sharpness:** mean predicted standard deviation and mean interval width, interpreted only
  alongside coverage.
- **Synthetic-only recovery:** RMSE and rank correlation between predicted and true `sigma(x)`.
- **Cost:** wall-clock training time, inference time and number of fitted networks.

Aggregate repeated runs with median and interquartile range plus mean and standard deviation.
Compare methods using paired per-split differences and bootstrap confidence intervals. Do not infer
superiority from overlapping or non-overlapping unpaired error bars alone.

## Stage 4: three genuinely new datasets

Dataset selection is frozen before viewing method results. Each dataset must be continuous-target
regression, legally redistributable or script-downloadable, large enough for repeated splits, and
scientifically meaningful. At least one should be space-physics or astronomy aligned. The final
three and exclusion criteria will be registered in a dated decision record before experiments begin.

## Known implementation decisions

- PyTorch is used for both neural methods so framework differences do not confound results.
- Variance is parameterized as `softplus(raw) + epsilon` for Deep Ensembles and `exp(log_sigma)`
  for paper-style ACCRUE.
- Deep Ensemble predictive variance uses the paper's moment-matching equation:
  `mean(member_variance + member_mean^2) - ensemble_mean^2`.
- The analytic ACCRUE reliability score sorts standardized errors inside each training batch.
  Full-batch optimization is therefore the default for ACCRUE.
- ACCRUE Algorithm 1 does not publish its polynomial stopping tolerance or positivity handling.
  We use a score-difference tolerance of `0.01` and reject candidate polynomials that are
  nonpositive on a 256-point grid spanning the known synthetic domain. This prevents invalid
  Gaussian scales between sparse training points and is reported as an implementation decision.
