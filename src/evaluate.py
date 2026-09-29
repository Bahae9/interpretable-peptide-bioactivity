"""Nested cross-validation: outer 5-fold estimate, inner 3-fold tuning."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

from src.features import COMPOSITION, GLOBAL, POSITIONAL, SCALES
from src.models import SEED, model_zoo

IN = Path("data/processed/features.csv")
OUT = Path("results/tables/nested_cv.csv")

META = ["sequence", "pIC50", "n_measurements", "fold_spread", "source"]


def load() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(IN)
    feats = [c for c in df.columns if c not in META]
    return df, df["pIC50"].to_numpy(), feats


def nested_cv(X, y, est, grid, n_outer=5, n_inner=3, seed=SEED):
    outer = KFold(n_splits=n_outer, shuffle=True, random_state=seed)
    rows = []
    for tr, te in outer.split(X):
        if grid:
            search = GridSearchCV(clone(est), grid,
                                  cv=KFold(n_inner, shuffle=True, random_state=seed),
                                  scoring="neg_root_mean_squared_error", n_jobs=-1)
            search.fit(X[tr], y[tr])
            best = search.best_estimator_
        else:
            best = clone(est).fit(X[tr], y[tr])
        pred = best.predict(X[te])
        rows.append({
            "r2": r2_score(y[te], pred),
            "rmse": float(np.sqrt(mean_squared_error(y[te], pred))),
            "spearman": spearmanr(y[te], pred).statistic if len(set(pred)) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def ablation(df, y, best_name):
    """Which descriptor block carries the signal?"""
    blocks = {
        "A: global + scales (46, mirrors lab)": GLOBAL + SCALES,
        "B: positional + composition (45)": POSITIONAL + COMPOSITION,
        "B1: positional only (25)": POSITIONAL,
        "A + B (91, full)": GLOBAL + SCALES + POSITIONAL + COMPOSITION,
    }
    est, grid = model_zoo()[best_name]
    rows = []
    print("\n" + "=" * 74)
    print(f"Feature-block ablation  ({best_name}, nested CV)")
    print("=" * 74)
    for name, cols in blocks.items():
        f = nested_cv(df[cols].to_numpy(dtype=float), y, est, grid)
        rows.append({"block": name, "n_features": len(cols),
                     "r2_mean": f.r2.mean(), "r2_std": f.r2.std(),
                     "spearman_mean": f.spearman.mean(),
                     "spearman_std": f.spearman.std()})
        print(f"{name:<40}{len(cols):>4}  R2 {f.r2.mean():>6.3f} +/-{f.r2.std():<6.3f}"
              f"  rho {f.spearman.mean():>6.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(Path("results/tables/ablation.csv"), index=False)
    return out


def main(shuffle_control: bool = True):
    df, y, feats = load()
    X = df[feats].to_numpy(dtype=float)
    print("=" * 74)
    print(f"Nested CV  (outer 5-fold / inner 3-fold, seed={SEED})   "
          f"n={len(y)}, p={len(feats)}")
    print("=" * 74)
    print(f"{'model':<24}{'R2':>18}{'RMSE':>16}{'Spearman':>16}")
    print("-" * 74)

    results = []
    for name, (est, grid) in model_zoo().items():
        f = nested_cv(X, y, est, grid)
        row = {"model": name}
        for m in ("r2", "rmse", "spearman"):
            row[f"{m}_mean"], row[f"{m}_std"] = f[m].mean(), f[m].std()
        results.append(row)
        print(f"{name:<24}"
              f"{row['r2_mean']:>10.3f} ±{row['r2_std']:<6.3f}"
              f"{row['rmse_mean']:>9.3f} ±{row['rmse_std']:<6.3f}"
              f"{row['spearman_mean']:>9.3f} ±{row['spearman_std']:<6.3f}")

    res = pd.DataFrame(results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    if shuffle_control:
        rng = np.random.default_rng(SEED)
        yshuf = rng.permutation(y)
        est, grid = model_zoo()["RandomForest"]
        f = nested_cv(X, yshuf, est, grid)
        print("-" * 74)
        print(f"{'RandomForest (y shuffled)':<24}{f.r2.mean():>10.3f} ±{f.r2.std():<6.3f}"
              "   <- leakage control, expect R2 ~ 0")
    best = res[res.model != "Baseline (mean)"]
    best_name = best.loc[best.r2_mean.idxmax(), "model"]
    ablation(df, y, best_name)

    print(f"\nwritten -> {OUT}")
    return res


if __name__ == "__main__":
    main()
