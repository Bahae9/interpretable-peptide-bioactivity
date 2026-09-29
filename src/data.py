"""Download and clean the AHTPDB ACE-inhibitory peptide dataset.

Source: Kumar et al. (2015) AHTPDB: a comprehensive platform for analysis and
presentation of antihypertensive peptides. Nucleic Acids Research 43:D956-65.
https://webs.iiitd.edu.in/raghava/ahtpdb/

The raw file is NOT redistributed in this repo; this module downloads it.
"""
from __future__ import annotations

import re
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://webs.iiitd.edu.in/raghava/ahtpdb/downloads/pepic50.txt"
RAW = Path("data/raw/pepic50.txt")
OUT = Path("data/processed/peptides_clean.csv")

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# AHTPDB mixes U+03BC (greek small letter mu) and U+00B5 (micro sign).
MICRO = {"μ": "u", "µ": "u"}


def download(force: bool = False) -> Path:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if force or not RAW.exists():
        urllib.request.urlretrieve(URL, RAW)
    return RAW


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).strip()
    for k, v in MICRO.items():
        s = s.replace(k, v)
    return s


def parse_ic50(raw: str, molwt: float) -> tuple[float | None, str]:
    """Parse one IC50 cell into micromolar. Returns (value_uM, status)."""
    s = _norm(raw)
    if not s or s.upper() in {"ND", "NA", "-"}:
        return None, "missing"
    if "%" in s:
        return None, "percent_unit"          # inhibition %, not a concentration
    censored = s.startswith((">", "<"))
    body = re.sub(r"^[><~=\s]*", "", s)
    m = re.match(r"([\d.,]+)\s*(?:[-–]\s*([\d.,]+))?\s*(.*)$", body)
    if not m:
        return None, "unparsed"
    try:
        val = float(m.group(1).replace(",", ""))
    except ValueError:
        return None, "unparsed"
    if m.group(2):                            # a range -> midpoint
        try:
            val = (val + float(m.group(2).replace(",", ""))) / 2
        except ValueError:
            pass
    if val <= 0:
        return None, "nonpositive"
    unit = m.group(3).lower().strip().rstrip(".")
    # normalise "uM/L", "uM/ml" -> "um"
    unit = re.split(r"[/(±]", unit)[0].strip()

    if unit.startswith("um") or unit == "":
        uM = val
    elif unit.startswith("mm"):
        uM = val * 1_000
    elif unit.startswith("nm"):
        uM = val / 1_000
    elif unit.startswith("ug") or unit.startswith("mcg"):
        if not np.isfinite(molwt) or molwt <= 0:
            return None, "mass_unit_no_mw"
        uM = val / molwt * 1_000            # ug/ml -> uM
    elif unit.startswith("mg"):
        if not np.isfinite(molwt) or molwt <= 0:
            return None, "mass_unit_no_mw"
        uM = val / molwt * 1_000_000        # mg/ml -> uM
    else:
        return None, "unknown_unit"
    return uM, ("censored" if censored else "ok")


def build() -> pd.DataFrame:
    path = download()
    df = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8",
                     engine="python", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    n_raw = len(df)

    molwt = pd.to_numeric(df["molwt"].map(_norm), errors="coerce")
    parsed = [parse_ic50(v, w) for v, w in zip(df["ic50"], molwt)]
    df["ic50_uM"] = [p[0] for p in parsed]
    df["status"] = [p[1] for p in parsed]
    df["seq"] = df["seq"].map(lambda s: _norm(s).upper())
    df["is_standard"] = df["seq"].map(lambda s: bool(s) and set(s) <= STANDARD_AA)

    reasons = df["status"].value_counts().to_dict()
    n_nonstd = int((~df["is_standard"]).sum())

    keep = df[(df["status"] == "ok") & df["is_standard"]].copy()
    keep["pIC50"] = -np.log10(keep["ic50_uM"] * 1e-6)

    # Aggregate replicate measurements of the same sequence.
    g = keep.groupby("seq")["pIC50"]
    out = pd.DataFrame({
        "sequence": g.median().index,
        "pIC50": g.median().values,
        "n_measurements": g.size().values,
        "pIC50_range": (g.max() - g.min()).values,
    })
    out["length"] = out["sequence"].str.len()
    # fold-spread in concentration space; 10**range because pIC50 is a log scale
    out["fold_spread"] = 10 ** out["pIC50_range"]
    src = keep.groupby("seq")["source"].agg(
        lambda s: sorted({_norm(x) for x in s if _norm(x).upper() != "ND"})[:1] or ["ND"])
    out["source"] = [s[0] for s in src.reindex(out["sequence"]).values]
    out = out.sort_values("pIC50", ascending=False).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("=" * 62)
    print("AHTPDB -> clean dataset")
    print("=" * 62)
    print(f"raw rows                     : {n_raw}")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  rows with status={k:<18}: {v}")
    print(f"rows with non-standard residues: {n_nonstd}")
    print(f"rows kept (usable, uncensored) : {len(keep)}")
    print(f"unique sequences               : {len(out)}")
    print(f"pIC50  min/median/max          : "
          f"{out.pIC50.min():.2f} / {out.pIC50.median():.2f} / {out.pIC50.max():.2f}")
    print(f"sequences with >1 measurement  : {(out.n_measurements > 1).sum()}")
    print(f"  of those, fold-spread p90    : "
          f"{out.loc[out.n_measurements > 1, 'fold_spread'].quantile(0.9):.1f}x")
    print(f"length min/median/max          : "
          f"{out.length.min()} / {int(out.length.median())} / {out.length.max()}")
    print(f"written -> {OUT}")
    return out


if __name__ == "__main__":
    build()
