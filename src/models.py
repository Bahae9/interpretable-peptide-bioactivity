"""Model zoo: penalised linear, ensemble, and Bayesian regressors."""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def _scaled(est) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", est)])


def model_zoo() -> dict[str, tuple[object, dict]]:
    """name -> (estimator, inner-CV parameter grid)."""
    return {
        "Baseline (mean)": (DummyRegressor(strategy="mean"), {}),
        "Lasso": (
            _scaled(Lasso(max_iter=50_000, random_state=SEED)),
            {"model__alpha": np.logspace(-3, 0, 12)},
        ),
        "ElasticNet": (
            _scaled(ElasticNet(max_iter=50_000, random_state=SEED)),
            {"model__alpha": np.logspace(-3, 0, 8),
             "model__l1_ratio": [0.15, 0.5, 0.85]},
        ),
        "BayesianRidge": (
            _scaled(BayesianRidge()),
            {"model__alpha_1": [1e-6, 1e-4], "model__lambda_1": [1e-6, 1e-4]},
        ),
        "RandomForest": (
            RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1),
            {"max_depth": [None, 12], "min_samples_leaf": [1, 3]},
        ),
        "HistGradientBoosting": (
            HistGradientBoostingRegressor(random_state=SEED),
            {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31],
             "min_samples_leaf": [10, 20]},
        ),
    }
