"""Classical ML baselines."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_model(name: str, seed: int = 42):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")
