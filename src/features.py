"""Compute physicochemical descriptors for each peptide.

Mirrors the ~45 descriptors used in the target lab's setting: 9 global
physicochemical properties plus five standard amino-acid scale families
(Kidera, Z-scales, VHSE, Cruciani, FASGAI, T-scales).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import peptides

IN = Path("data/processed/peptides_clean.csv")
OUT = Path("data/processed/features.csv")

# Scale families taken from peptides.Peptide.descriptors(); interpretable and
# widely used in peptide QSAR.
SCALES = (
    [f"KF{i}" for i in range(1, 11)]      # Kidera factors
    + [f"Z{i}" for i in range(1, 6)]      # Z-scales (Hellberg)
    + [f"VHSE{i}" for i in range(1, 9)]   # VHSE
    + [f"PP{i}" for i in range(1, 4)]     # Cruciani properties
    + [f"F{i}" for i in range(1, 7)]      # FASGAI
    + [f"T{i}" for i in range(1, 6)]      # T-scales
)

GLOBAL = [
    "length", "molecular_weight", "charge", "isoelectric_point",
    "hydrophobicity", "hydrophobic_moment", "aliphatic_index",
    "instability_index", "boman",
]

AA = "ACDEFGHIKLMNPQRSTVWY"

# Positions that matter for ACE inhibition: the C-terminal tripeptide is the
# classic determinant (Pro/Trp/Tyr/Phe at the C-terminus give potent
# inhibitors), plus the two N-terminal residues. Whole-sequence averages
# cannot express "the last residue is tryptophan", so these are encoded
# explicitly as per-residue Z-scale values.
POSITIONS = [("Ct", -1), ("Ct1", -2), ("Ct2", -3), ("Nt", 0), ("Nt1", 1)]

_ZTAB = {a: [peptides.Peptide(a * 3).descriptors()[f"Z{i}"] for i in range(1, 6)]
         for a in AA}

POSITIONAL = [f"{tag}_Z{i}" for tag, _ in POSITIONS for i in range(1, 6)]
COMPOSITION = [f"frac_{a}" for a in AA]


def positional(seq: str) -> dict[str, float]:
    row = {}
    for tag, idx in POSITIONS:
        aa = seq[idx] if -len(seq) <= idx < len(seq) else None
        vals = _ZTAB[aa] if aa else [0.0] * 5
        for i, v in enumerate(vals, 1):
            row[f"{tag}_Z{i}"] = v
    return row


def composition(seq: str) -> dict[str, float]:
    return {f"frac_{a}": seq.count(a) / len(seq) for a in AA}


def describe(seq: str) -> dict[str, float]:
    p = peptides.Peptide(seq)
    d = p.descriptors()
    row: dict[str, float] = {
        "length": float(len(seq)),
        "molecular_weight": p.molecular_weight(),
        "charge": p.charge(pH=7.0),
        "isoelectric_point": p.isoelectric_point(),
        "hydrophobicity": p.hydrophobicity(),
        "hydrophobic_moment": p.hydrophobic_moment() or 0.0,
        "aliphatic_index": p.aliphatic_index(),
        "instability_index": p.instability_index(),
        "boman": p.boman(),
    }
    for k in SCALES:
        row[k] = d.get(k, np.nan)
    row.update(positional(seq))
    row.update(composition(seq))
    return row


def build() -> pd.DataFrame:
    df = pd.read_csv(IN)
    feats = pd.DataFrame([describe(s) for s in df["sequence"]])
    cols = GLOBAL + SCALES + POSITIONAL + COMPOSITION
    feats = feats[cols]

    n_nan = int(feats.isna().sum().sum())
    n_inf = int(np.isinf(feats.to_numpy(dtype=float)).sum())

    out = pd.concat([df[["sequence", "pIC50", "n_measurements",
                         "fold_spread", "source"]], feats], axis=1)
    out.to_csv(OUT, index=False)

    print("=" * 62)
    print("descriptors")
    print("=" * 62)
    print(f"peptides                 : {len(out)}")
    print(f"descriptors per peptide  : {len(cols)}")
    print(f"  block A  global          : {len(GLOBAL)}")
    print(f"  block A  scale-family    : {len(SCALES)}   "
          f"(-> {len(GLOBAL) + len(SCALES)} mirroring the lab's 45)")
    print(f"  block B  positional      : {len(POSITIONAL)}")
    print(f"  block B  composition     : {len(COMPOSITION)}")
    print(f"NaN cells / inf cells    : {n_nan} / {n_inf}")
    if n_nan:
        print("  columns with NaN:", feats.columns[feats.isna().any()].tolist())
    zero_var = [c for c in cols if feats[c].std(ddof=0) == 0]
    print(f"zero-variance descriptors : {len(zero_var)} {zero_var}")
    print(f"written -> {OUT}")
    return out


if __name__ == "__main__":
    build()
