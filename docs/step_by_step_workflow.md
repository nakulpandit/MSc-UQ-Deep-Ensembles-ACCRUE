# Step-by-step reproduction workflow

This workflow is designed for completing one research step, checking it, and
pushing it to GitHub before beginning the next step.

## One-time environment setup

From the repository root on macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

What this does:

1. `.venv` isolates this project's packages from the rest of your Mac.
2. Activating it makes `python`, `pytest` and the `uq-*` commands use that environment.
3. Editable installation makes changes under `src/` immediately available.
4. `pytest` checks the equations and data pipeline before expensive experiments.

Run `source .venv/bin/activate` again whenever you open a new Terminal window.

## Step 1: G reproduction

Why: G is the simplest ACCRUE heteroscedastic dataset. Its true uncertainty is
linear, so it is the best first check of whether the variance-recovery pipeline works.

```bash
git switch -c results/accrue-polynomial-g
uq-reproduce-accrue --dataset g --runs 100
uq-validate-results results/paper_reproductions/accrue_polynomial_g \
  --dataset g --runs 100
```

Inspect:

```bash
open results/paper_reproductions/accrue_polynomial_g/sigma_recovery.png
open results/paper_reproductions/accrue_polynomial_g/reliability_diagram.png
cat results/paper_reproductions/accrue_polynomial_g/summary.json
```

Commit and push:

```bash
git add results/paper_reproductions/accrue_polynomial_g
git commit -m "Add 100-run ACCRUE polynomial G results"
git push -u origin results/accrue-polynomial-g
```

Open a pull request on GitHub and merge it into `main` after reviewing the four files.

## Step 2: Y reproduction

Why: Y has a more nonlinear mean and non-monotonic noise. It checks whether
ACCRUE can recover uncertainty that cannot be represented by a simple straight line.

First update local `main` after merging Step 1:

```bash
git switch main
git pull
git switch -c results/accrue-polynomial-y
```

Run, validate and inspect:

```bash
uq-reproduce-accrue --dataset y --runs 100
uq-validate-results results/paper_reproductions/accrue_polynomial_y \
  --dataset y --runs 100
open results/paper_reproductions/accrue_polynomial_y/sigma_recovery.png
open results/paper_reproductions/accrue_polynomial_y/reliability_diagram.png
cat results/paper_reproductions/accrue_polynomial_y/summary.json
```

Commit and push:

```bash
git add results/paper_reproductions/accrue_polynomial_y
git commit -m "Add 100-run ACCRUE polynomial Y results"
git push -u origin results/accrue-polynomial-y
```

Open, review and merge the pull request before Step 3.

## Step 3: W reproduction

Why: W has strongly varying noise and is the hardest one-dimensional polynomial
case in the ACCRUE paper. It tests the limitation of a smooth polynomial variance model.

```bash
git switch main
git pull
git switch -c results/accrue-polynomial-w
uq-reproduce-accrue --dataset w --runs 100
uq-validate-results results/paper_reproductions/accrue_polynomial_w \
  --dataset w --runs 100
open results/paper_reproductions/accrue_polynomial_w/sigma_recovery.png
open results/paper_reproductions/accrue_polynomial_w/reliability_diagram.png
cat results/paper_reproductions/accrue_polynomial_w/summary.json
```

Commit and push:

```bash
git add results/paper_reproductions/accrue_polynomial_w
git commit -m "Add 100-run ACCRUE polynomial W results"
git push -u origin results/accrue-polynomial-w
```

Open, review and merge the pull request.

## About the quick neural command

```bash
uq-reproduce-toys --dataset g --method both --runs 1 --quick
```

This is a development smoke test for the PyTorch Deep Ensemble and neural ACCRUE
implementations. It does not reproduce a paper table and should not be committed
as a final result. Run it after `pytest`, but treat its output as a diagnostic only.

## Reproducing versus downloading results

The downloadable reference result archive is for comparison and recovery. Your
main evidence should be generated on your own machine with the commands above.
If your metrics differ slightly, that can be normal across operating systems and
library versions. Seeds, row counts, file structure and qualitative plots should match.
