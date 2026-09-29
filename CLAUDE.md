# Project: Interpretable Prediction of Peptide Bioactivity

## Why this project exists
I'm Bahae (Lemtai Bahaeddine), a Master's student in Bioinformatics (Faculty of Computer Science,
USTHB, Algeria) with an AI-engineering background (PyTorch, LLMs, OCR). I'm applying for a 6-month
M2 internship, "AI x Bioactive Peptides", run jointly by Sorbonne Université (LPSM / SCAI,
Rafael Pinot) and Université Laval (Laurent Bazinet), starting Feb/March 2027.

Their task: about 300 peptides (sequences + 45 physicochemical descriptors), with bioactivity
measured on **fractions** (mixtures), not on individual peptides. They use penalized linear models,
Random Forest, Gradient Boosting and Bayesian models; SHAP and stability selection; and output
a shortlist of 10–20 candidate peptides for synthesis.

This repo is a small, public **mirror of their problem**, built to prove I can do the job.
It goes on my CV and GitHub (pinned) and gets linked in my application email.

**Time budget: 5–7 days.** The job posting is already a month old, so a clean MVP sent soon beats
a perfect repo sent late.

## Goals, in priority order
1. **Data**: a few hundred food-derived peptides from a public database, labelled for one
   activity. Candidate sources: BIOPEP-UWM (ACE-inhibitory / antioxidant; closest to Bazinet's
   food-peptide work), DBAASP or APD3 (antimicrobial). Pick ONE activity and document the choice.
   Verify each database's access method and license before scraping or downloading.
2. **Features**: about 40–50 physicochemical descriptors per peptide, to mirror their 45
   (candidate tools: `peptides` or `modlAMP`; Biopython ProtParam as a fallback).
3. **Models**: Lasso / Elastic Net, Random Forest, Gradient Boosting, and one Bayesian model.
4. **Rigor**:
   - Nested cross-validation, reporting mean ± std across outer folds, with fixed seeds.
   - Handle and discuss correlated descriptors: SHAP and Lasso are unstable under collinearity,
     and stability selection helps with that. This is a likely interview topic, so the README
     must explain it.
5. **Interpretation**: SHAP plots, stability-selection results, and a ranked shortlist of about 15
   candidate peptides.
6. **The standout experiment**: simulate fraction-level (mixture) labels. Pool peptides into
   synthetic fractions and give each fraction a mixture-level label (e.g., the max or a
   weighted sum of member activities). Then test whether truly active peptides can be recovered
   (e.g., top-k recall). Frame it as multiple-instance learning / attribution. Even a modest
   result is valuable.
7. **README as a mini-paper**: problem, data, method, results table, 2–3 figures, limitations,
   and how to reproduce.

## Working rules (important)
- **Consult me before adding heavy dependencies or expanding scope.** Pre-approved: scikit-learn,
  pandas, numpy, matplotlib/seaborn, shap, and one of `peptides` / `modlAMP`. Ask before adding
  PyMC, XGBoost/LightGBM, or anything else heavy (sklearn's BayesianRidge and
  HistGradientBoosting are acceptable defaults).
- **Validate against the real environment.** Run the code and check the outputs; don't assume
  correctness. Report actual numbers only, never invented ones.
- Keep it small, reproducible, and readable: a reviewer skims it in a few minutes.

## Suggested structure
```
data/            raw + processed (or a download script if the license forbids redistribution)
src/             data.py, features.py, models.py, evaluate.py, mixtures.py
notebooks/       one clean walkthrough notebook
results/         figures + tables
README.md
requirements.txt
```

## CV placeholders this project must fill (keep a list of the real values)
- Repo name and date
- N peptides, source database, activity chosen, number of descriptors, descriptor tool
- Bayesian model type
- Best model + metric (AUC or R²) as mean ± std under nested CV
- Shortlist size
- Mixture-recovery result (e.g., top-20 recall)
