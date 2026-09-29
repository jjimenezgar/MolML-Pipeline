"""Command-line training entry point for V1."""

from __future__ import annotations

import argparse
import json

from molml.data import load_bace
from molml.evaluate import classification_metrics
from molml.features import fingerprint_matrix
from molml.models import build_model
from molml.splitting import random_split, scaffold_split


def run(data_path: str, split_name: str, model_name: str, seed: int = 42):
    df = load_bace(data_path)
    splitter = scaffold_split if split_name == "scaffold" else random_split
    splits = splitter(df) if split_name == "scaffold" else splitter(df, seed=seed)
    train, val, test = splits

    x_train = fingerprint_matrix(train["smiles"].tolist())
    x_val = fingerprint_matrix(val["smiles"].tolist())
    x_test = fingerprint_matrix(test["smiles"].tolist())

    model = build_model(model_name, seed=seed)
    model.fit(x_train, train["label"])

    results = {
        "dataset_size": len(df),
        "class_balance": df["label"].value_counts().sort_index().to_dict(),
        "split": split_name,
        "model": model_name,
        "sizes": {"train": len(train), "validation": len(val), "test": len(test)},
    }

    for name, frame, x in (("validation", val, x_val), ("test", test, x_test)):
        pred = model.predict(x)
        prob = model.predict_proba(x)[:, 1]
        results[name] = classification_metrics(frame["label"], pred, prob)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train a MolML-Pipeline V1 baseline.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=["random", "scaffold"], default="scaffold")
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "random_forest"],
        default="random_forest",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.split, args.model, args.seed), indent=2))


if __name__ == "__main__":
    main()
