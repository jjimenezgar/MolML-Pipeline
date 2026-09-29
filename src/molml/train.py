"""Command-line training entry point for V1."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from molml.config import ExperimentConfig, load_config
from molml.data import load_bace
from molml.evaluate import classification_metrics
from molml.features import fingerprint_matrix
from molml.models import build_model
from molml.plots import save_test_plots
from molml.splitting import bemis_murcko_scaffold, random_split, scaffold_split


def _counts(frame):
    return {str(i): int((frame["label"] == i).sum()) for i in (0, 1)}


def run_experiment(config: ExperimentConfig, output_dir: Path | None = None):
    data_path = Path(config.data_path)
    df = load_bace(data_path)
    if df["label"].nunique() != 2:
        raise ValueError("BACE requires two classes.")
    splitter = scaffold_split if config.split == "scaffold" else random_split
    train, val, test = splitter(df, train_fraction=config.train_fraction,
                                val_fraction=config.val_fraction, seed=config.seed)
    parts = {"train": train, "validation": val, "test": test}
    scaffold_sets = {name: {bemis_murcko_scaffold(s) for s in part["smiles"]}
                     for name, part in parts.items()}
    overlaps = {"train_validation": len(scaffold_sets["train"] & scaffold_sets["validation"]),
                "train_test": len(scaffold_sets["train"] & scaffold_sets["test"]),
                "validation_test": len(scaffold_sets["validation"] & scaffold_sets["test"])}
    if any(part["label"].nunique() != 2 for part in parts.values()):
        raise ValueError("A partition lacks one class; evaluation is invalid.")
    if config.split == "scaffold" and any(overlaps.values()):
        raise ValueError("Scaffold leakage detected.")

    def features(frame):
        return fingerprint_matrix(frame["smiles"].tolist(), radius=config.radius, n_bits=config.n_bits)

    model = build_model(config.model, seed=config.seed, parameters=config.model_parameters)
    model.fit(features(train), train["label"])
    result = {
        "config": asdict(config),
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "dataset_size": len(df),
        "class_balance": _counts(df),
        "partitions": {name: {"size": len(part), "class_balance": _counts(part),
                              "unique_scaffolds": len(scaffold_sets[name])}
                       for name, part in parts.items()},
        "scaffold_overlap": overlaps,
        "threshold": 0.5,
        "versions": _versions(),
    }
    for name in ("validation", "test"):
        part = parts[name]
        probabilities = model.predict_proba(features(part))[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        result[name] = classification_metrics(part["label"], predictions, probabilities)
        if name == "test" and output_dir is not None:
            save_test_plots(part["label"], predictions, probabilities, output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def _versions():
    import numpy, pandas, rdkit, sklearn, matplotlib, yaml
    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "pandas": pandas.__version__, "rdkit": rdkit.__version__,
            "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
            "pyyaml": yaml.__version__}


def run(data_path: str, split_name: str, model_name: str, seed: int = 42):
    """Backwards-compatible programmatic entry point with V1 defaults."""
    params = ({"max_iter": 2000, "class_weight": "balanced"} if model_name == "logistic_regression"
              else {"n_estimators": 300, "class_weight": "balanced", "n_jobs": 1})
    cfg = ExperimentConfig(model_name, split_name, seed, 2, 2048, 0.8, 0.1, 0.1, params, data_path)
    result = run_experiment(cfg)
    return {"dataset_size": result["dataset_size"], "class_balance": result["class_balance"],
            "split": split_name, "model": model_name,
            "sizes": {name: part["size"] for name, part in result["partitions"].items()},
            "validation": result["validation"], "test": result["test"]}


def main():
    parser = argparse.ArgumentParser(description="Train a MolML-Pipeline V1 baseline.")
    parser.add_argument("--config", type=Path, help="YAML experiment configuration")
    parser.add_argument("--output", type=Path, help="Directory for metrics and test plots")
    parser.add_argument("--data", help="Legacy dataset path (use --config for reproducible runs)")
    parser.add_argument("--split", choices=["random", "scaffold"], default="scaffold")
    parser.add_argument("--model", choices=["logistic_regression", "random_forest"], default="random_forest")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.config and args.data or not args.config and not args.data:
        parser.error("Provide exactly one of --config or --data.")
    if args.config:
        cfg = load_config(args.config)
        result = run_experiment(cfg, args.output)
    else:
        if args.output:
            parser.error("--output requires --config.")
        result = run(args.data, args.split, args.model, args.seed)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
