"""First executable check: fit both uncertainty models on an ACCRUE toy generator."""

import argparse
import json

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from uqcomparison.data import generate_synthetic
from uqcomparison.metrics import calibration_error, gaussian_crps, gaussian_nll, rmse


def _split(sample, seed: int):
    indices = np.arange(len(sample.y))
    train, remainder = train_test_split(indices, train_size=0.33, random_state=seed)
    validation, test = train_test_split(
        remainder,
        train_size=0.33 / 0.67,
        random_state=seed + 1,
    )
    return train, validation, test


def _scores(y, mean, sigma, true_sigma):
    return {
        "rmse": rmse(y, mean),
        "nll": gaussian_nll(y, mean, sigma),
        "crps": gaussian_crps(y, mean, sigma),
        "calibration_error": calibration_error(y, mean, sigma),
        "sigma_rmse": rmse(true_sigma, sigma),
        "mean_sigma": float(np.mean(sigma)),
    }


def run_once(dataset: str, method: str, seed: int, quick: bool):
    from uqcomparison.models.accrue import AccrueConfig, AccrueVarianceRegressor
    from uqcomparison.models.deep_ensemble import DeepEnsembleConfig, DeepEnsembleRegressor

    sample = generate_synthetic(dataset, seed=seed)
    train, validation, test = _split(sample, seed)
    x_scaler = StandardScaler().fit(sample.x[train])
    x = x_scaler.transform(sample.x)
    output = {}

    if method in {"accrue", "both"}:
        if dataset.lower() in {"5d", "five_d"}:
            oracle_mean = sample.mean
        else:
            gp = GaussianProcessRegressor(
                kernel=RBF() + WhiteKernel(), normalize_y=True, random_state=seed
            ).fit(x[train], sample.y[train])
            oracle_mean = gp.predict(x)
        residual = sample.y - oracle_mean
        config = AccrueConfig(
            restarts=1 if quick else 5,
            max_epochs=3 if quick else 100,
            lbfgs_iterations_per_epoch=2 if quick else 5,
        )
        accrue = AccrueVarianceRegressor(x.shape[1], config=config, seed=seed).fit(
            x[train], residual[train], x[validation], residual[validation]
        )
        sigma = accrue.predict_sigma(x[test])
        output["accrue"] = _scores(
            sample.y[test], oracle_mean[test], sigma, sample.sigma[test]
        )

    if method in {"deep_ensemble", "both"}:
        config = DeepEnsembleConfig(
            members=2 if quick else 5,
            epochs=5 if quick else 40,
        )
        ensemble = DeepEnsembleRegressor(x.shape[1], config=config, seed=seed).fit(
            x[train], sample.y[train]
        )
        mean, sigma = ensemble.predict(x[test])
        output["deep_ensemble"] = _scores(
            sample.y[test], mean, sigma, sample.sigma[test]
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["g", "y", "w", "5d"], default="g")
    parser.add_argument("--method", choices=["accrue", "deep_ensemble", "both"], default="both")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    results = [
        run_once(args.dataset, args.method, args.seed + run, args.quick)
        for run in range(args.runs)
    ]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
