"""Fraction-level (mixture) labels: can active peptides be recovered?

This mirrors the target lab's real constraint. Bioactivity is assayed on
chromatographic *fractions* (mixtures of many peptides), not on individual
peptides, yet the deliverable is a shortlist of individual peptides to
synthesise. We simulate that: peptides are pooled into synthetic fractions,
only the fraction-level label is exposed to the model, and we measure how well
the truly active individual peptides are recovered.

Framed as multiple-instance learning: a fraction is a bag, its label is driven
by its most active member.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.evaluate import load

TAB = Path("results/tables")
SEED = 42
ACTIVE_Q = 0.90          # "active" = top decile of true pIC50


def make_fractions(n, k, rng):
    idx = rng.permutation(n)
    return [idx[i:i + k] for i in range(0, n, k)]


def bag_label(y, members, rule):
    v = y[members]
    if rule == "max":
        return v.max()
    if rule == "weighted_sum":
        # potency adds in concentration space, then back to log scale
        return np.log10(np.sum(10 ** v) )
    raise ValueError(rule)


def fit_propagate(X, y_bag, frac_of):
    """Baseline: every member inherits its fraction's label."""
    m = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    m.fit(X, y_bag[frac_of])
    return m.predict(X)


def fit_mil_max(X, y_bag, fractions, frac_of, n_iter=6):
    """MIL with max-pooling, by alternating optimisation.

    Each round the model picks the 'witness' in each fraction (its arg-max
    member); that witness carries the fraction label at full weight, the rest
    are down-weighted. This encodes 'one active peptide drives the fraction'.
    """
    y_inst = y_bag[frac_of].astype(float)
    w = np.ones(len(X))
    scores = np.zeros(len(X))
    for _ in range(n_iter):
        m = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        m.fit(X, y_inst, sample_weight=w)
        scores = m.predict(X)
        w = np.full(len(X), 0.1)
        for members in fractions:
            win = members[np.argmax(scores[members])]
            w[win] = 1.0
    return scores


def recall_at_k(scores, is_active, k):
    top = np.argsort(-scores)[:k]
    return is_active[top].sum() / is_active.sum()


def main(k_sizes=(5, 10, 20), rules=("max", "weighted_sum"), n_repeat=5):
    df, y, feats = load()
    X = df[feats].to_numpy(dtype=float)
    n = len(y)
    thr = np.quantile(y, ACTIVE_Q)
    is_active = (y >= thr).astype(int)
    n_act = int(is_active.sum())
    K = n_act                       # retrieve as many as there are actives

    print("=" * 78)
    print("Fraction-level (mixture) label recovery")
    print("=" * 78)
    print(f"peptides {n}   actives (top decile, pIC50>={thr:.2f}) = {n_act}   "
          f"recall measured at top-{K}")
    print(f"random-baseline recall = {K / n:.3f}\n")
    # Reference ceiling: the same model trained on INDIVIDUAL labels, scored
    # out-of-fold. Fraction-level results should be read against this, not
    # against a perfect oracle.
    from src.interpret import out_of_fold
    ceil = recall_at_k(out_of_fold(X, y, "RandomForest"), is_active, K)
    print(f"individual-label ceiling (OOF RandomForest) recall = {ceil:.3f}\n")

    print(f"{'rule':<14}{'bag size':>9}{'propagate':>22}{'MIL max-pool':>22}")
    print("-" * 78)

    rows = []
    for rule in rules:
        for k in k_sizes:
            rp, rm = [], []
            for rep in range(n_repeat):
                rng = np.random.default_rng(SEED + rep)
                fractions = make_fractions(n, k, rng)
                frac_of = np.full(n, -1)
                y_bag = np.zeros(len(fractions))
                for b, members in enumerate(fractions):
                    frac_of[members] = b
                    y_bag[b] = bag_label(y, members, rule)
                s_prop = fit_propagate(X, y_bag, frac_of)
                s_mil = fit_mil_max(X, y_bag, fractions, frac_of)
                rp.append(recall_at_k(s_prop, is_active, K))
                rm.append(recall_at_k(s_mil, is_active, K))
            rows.append({"rule": rule, "bag_size": k,
                         "propagate_mean": np.mean(rp), "propagate_std": np.std(rp),
                         "mil_mean": np.mean(rm), "mil_std": np.std(rm),
                         "random": K / n, "individual_label_ceiling": ceil})
            print(f"{rule:<14}{k:>9}"
                  f"{np.mean(rp):>14.3f} ±{np.std(rp):<6.3f}"
                  f"{np.mean(rm):>14.3f} ±{np.std(rm):<6.3f}")

    out = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    out.to_csv(TAB / "mixture_recovery.csv", index=False)
    best = out.loc[out.mil_mean.idxmax()]
    print("-" * 78)
    print(f"best MIL recall = {best.mil_mean:.3f} (rule={best['rule']}, "
          f"bag={int(best.bag_size)}) vs random {K / n:.3f}  "
          f"-> {best.mil_mean / (K / n):.1f}x enrichment")
    print(f"that recovers {best.mil_mean / ceil:.0%} of the individual-label "
          f"ceiling ({ceil:.3f}) while never seeing an individual label")
    print(f"\nwritten -> {TAB}/mixture_recovery.csv")
    return out


if __name__ == "__main__":
    main()
