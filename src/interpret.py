"""Interpretation: SHAP, stability selection, and a validated shortlist.

The shortlist is built from out-of-fold predictions, so every peptide is ranked
by a model that never saw it. Its true measured pIC50 is shown alongside, which
makes the shortlist verifiable rather than merely asserted.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.linear_model import Lasso, lasso_path
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

from src.evaluate import load
from src.models import SEED, model_zoo

FIG = Path("results/figures")
TAB = Path("results/tables")


def out_of_fold(X, y, name, n_outer=5):
    """OOF predictions from the outer CV loop."""
    est, grid = model_zoo()[name]
    oof = np.zeros(len(y))
    for tr, te in KFold(n_outer, shuffle=True, random_state=SEED).split(X):
        if grid:
            s = GridSearchCV(clone(est), grid, cv=KFold(3, shuffle=True,
                             random_state=SEED),
                             scoring="neg_root_mean_squared_error", n_jobs=-1)
            s.fit(X[tr], y[tr]); m = s.best_estimator_
        else:
            m = clone(est).fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def stability_selection(X, y, feats, n_boot=500, frac=0.75, n_alphas=40):
    """Bootstrap Lasso: how often is each descriptor selected across resamples
    and across the regularisation path? Robust where raw Lasso is not."""
    rng = np.random.default_rng(SEED)
    Xs = StandardScaler().fit_transform(X)
    n = len(y)
    counts = np.zeros(len(feats))
    for _ in range(n_boot):
        idx = rng.choice(n, size=int(frac * n), replace=False)
        _, coefs, _ = lasso_path(Xs[idx], y[idx], n_alphas=n_alphas, eps=1e-2)
        counts += (np.abs(coefs) > 1e-8).any(axis=1).astype(float)
    return pd.DataFrame({"descriptor": feats,
                         "selection_freq": counts / n_boot}
                        ).sort_values("selection_freq", ascending=False)


def main(best_name: str | None = None):
    df, y, feats = load()
    X = df[feats].to_numpy(dtype=float)
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

    if best_name is None:
        res = pd.read_csv(TAB / "nested_cv.csv")
        res = res[res.model != "Baseline (mean)"]
        best_name = res.loc[res.r2_mean.idxmax(), "model"]
    print(f"best model by nested-CV R2: {best_name}")

    # ---- 1. correlation structure -------------------------------------
    corr = pd.DataFrame(X, columns=feats).corr().abs()
    iu = np.triu_indices_from(corr, k=1)
    n_high = int((corr.to_numpy()[iu] > 0.9).sum())
    print(f"descriptor pairs with |r| > 0.9 : {n_high} of {len(iu[0])}")
    g = sns_heat(corr)
    g.savefig(FIG / "descriptor_correlation.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # ---- 2. SHAP ------------------------------------------------------
    est, grid = model_zoo()[best_name]
    if grid:
        s = GridSearchCV(clone(est), grid, cv=KFold(3, shuffle=True, random_state=SEED),
                         scoring="neg_root_mean_squared_error", n_jobs=-1).fit(X, y)
        model = s.best_estimator_
    else:
        model = clone(est).fit(X, y)

    inner = model.named_steps["model"] if hasattr(model, "named_steps") else model
    Xs = (model.named_steps["scale"].transform(X)
          if hasattr(model, "named_steps") else X)
    if hasattr(inner, "estimators_") or "Boosting" in type(inner).__name__:
        expl = shap.TreeExplainer(inner)
        sv = expl.shap_values(Xs)
    else:
        expl = shap.LinearExplainer(inner, Xs)
        sv = expl.shap_values(Xs)

    shap.summary_plot(sv, pd.DataFrame(Xs, columns=feats), show=False, max_display=20)
    plt.title(f"SHAP — {best_name}")
    plt.savefig(FIG / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    shap_imp = pd.DataFrame({"descriptor": feats,
                             "mean_abs_shap": np.abs(sv).mean(axis=0)}
                            ).sort_values("mean_abs_shap", ascending=False)

    # ---- 3. stability selection ---------------------------------------
    stab = stability_selection(X, y, feats)
    merged = shap_imp.merge(stab, on="descriptor")
    merged["shap_rank"] = merged.mean_abs_shap.rank(ascending=False).astype(int)
    merged["stability_rank"] = merged.selection_freq.rank(ascending=False,
                                                          method="min").astype(int)
    merged.to_csv(TAB / "interpretation.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    top = stab.head(20).iloc[::-1]
    ax.barh(top.descriptor, top.selection_freq, color="#4C72B0")
    ax.axvline(0.6, ls="--", c="crimson", lw=1, label="threshold 0.6")
    ax.set_xlabel("selection frequency (500 bootstrap Lasso fits)")
    ax.set_title("Stability selection"); ax.legend()
    fig.savefig(FIG / "stability_selection.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # ---- 4. validated shortlist ---------------------------------------
    oof = out_of_fold(X, y, best_name)
    df = df.assign(pred_oof=oof)
    short = df.nlargest(15, "pred_oof")[
        ["sequence", "length", "pred_oof", "pIC50", "n_measurements",
         "fold_spread", "source"]] if "length" in df.columns else \
        df.nlargest(15, "pred_oof")[
            ["sequence", "pred_oof", "pIC50", "n_measurements", "fold_spread", "source"]]
    short.to_csv(TAB / "shortlist.csv", index=False)

    top_q = df.pIC50.quantile(0.90)
    hit = (short.pIC50 >= top_q).mean()
    print("\n" + "=" * 68)
    print("Interpretation")
    print("=" * 68)
    print(f"|r|>0.9 descriptor pairs      : {n_high}")
    print(f"top-5 by SHAP                 : {shap_imp.descriptor.head(5).tolist()}")
    print(f"top-5 by stability selection  : {stab.descriptor.head(5).tolist()}")
    print(f"descriptors with freq >= 0.6  : {(stab.selection_freq >= 0.6).sum()}")
    print(f"\nshortlist (15) mean true pIC50 : {short.pIC50.mean():.2f} "
          f"(dataset mean {df.pIC50.mean():.2f})")
    print(f"shortlist precision @ top-decile (pIC50>={top_q:.2f}) : {hit:.0%} "
          f"(random baseline 10%)")
    print(f"\nwritten -> {TAB}/interpretation.csv, {TAB}/shortlist.csv")
    return merged, short


def sns_heat(corr):
    import seaborn as sns
    g = sns.clustermap(corr, cmap="viridis", figsize=(11, 10),
                       xticklabels=True, yticklabels=True,
                       cbar_kws={"label": "|Pearson r|"})
    g.ax_heatmap.tick_params(labelsize=6)
    return g


if __name__ == "__main__":
    main()
