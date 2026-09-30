# Changelog

## 1.2.0 — 2026-09-30

- Reproducible BACE baseline experiments with Morgan fingerprints, Logistic Regression and Random Forest, random and disjoint scaffold partitions, metrics and plots.
- Training-only Tanimoto similarity diagnostics, scaffold novelty and optional local MLflow tracking.
- SHAP attribution for eight fixed test molecules per scaffold model, with explicit output units and numerical additivity checks.
- Direct benchmark dependency constraints, Python 3.12 benchmark CI, and MLflow persistence checks alongside Python 3.10/3.11 compatibility checks.
- Updated project purpose, completed stages and limitations in the README.

Predictions are unvalidated model outputs. Similarity coverage is heuristic; hashed fingerprint attributions do not establish biological mechanisms. The benchmark uses one seeded partition per split type and does not measure variation across partitions.
