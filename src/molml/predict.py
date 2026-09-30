"""Local batch inference from a saved MolML model bundle."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from molml.artifacts import load_model_bundle
from molml.features import fingerprint_matrix, parse_smiles
from molml.splitting import bemis_murcko_scaffold


OUTPUT_FIELDS = ["prediction", "probability_class_1", "nearest_train_similarity",
                 "in_similarity_domain", "novel_scaffold", "status", "error"]


def predict_smiles(bundle: dict, smiles_list: list[str]) -> list[dict]:
    """Predict class-1 scores and training-domain diagnostics for each SMILES.

    Invalid entries remain in the output with an error, so one bad molecule
    does not discard results for the rest of a batch.
    """
    fp_cfg = bundle["fingerprint"]
    radius, n_bits = int(fp_cfg["radius"]), int(fp_cfg["n_bits"])
    threshold = float(bundle["decision_threshold"])
    domain_threshold = float(bundle["similarity_domain"]["threshold"])
    train_fp = np.asarray(bundle["training_fingerprints"], dtype=np.uint8)
    train_sums = train_fp.sum(axis=1, dtype=np.int64)
    train_scaffolds = set(bundle["training_scaffolds"])
    records = [{"smiles": value, "prediction": "", "probability_class_1": "",
                "nearest_train_similarity": "", "in_similarity_domain": "",
                "novel_scaffold": "", "status": "invalid", "error": ""}
               for value in smiles_list]
    valid_indices, valid_smiles = [], []
    for index, value in enumerate(smiles_list):
        try:
            parse_smiles(value)
        except ValueError as exc:
            records[index]["error"] = str(exc)
        else:
            valid_indices.append(index)
            valid_smiles.append(value)
    if not valid_smiles:
        return records

    query_fp = fingerprint_matrix(valid_smiles, radius=radius, n_bits=n_bits).astype(np.uint8)
    model = bundle["model"]
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Saved model does not contain the positive class (1).")
    probabilities = model.predict_proba(query_fp)[:, classes.index(1)]
    intersections = query_fp.astype(np.int64) @ train_fp.T.astype(np.int64)
    unions = query_fp.sum(axis=1, dtype=np.int64)[:, None] + train_sums[None, :] - intersections
    similarities = np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions > 0)
    nearest = similarities.max(axis=1)
    novel = [bemis_murcko_scaffold(value) not in train_scaffolds for value in valid_smiles]
    for index, probability, similarity, is_novel in zip(valid_indices, probabilities, nearest, novel):
        records[index].update({
            "prediction": int(probability >= threshold),
            "probability_class_1": float(probability),
            "nearest_train_similarity": float(similarity),
            "in_similarity_domain": bool(similarity >= domain_threshold),
            "novel_scaffold": bool(is_novel),
            "status": "ok",
        })
    return records


def _read_input(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "smiles" not in reader.fieldnames:
            raise ValueError("Input CSV must contain a column named 'smiles'.")
        return list(reader), list(reader.fieldnames)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict BACE benchmark scores for new SMILES.")
    parser.add_argument("--model", required=True, type=Path, help="Trusted model bundle created by molml.train")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="CSV file with a 'smiles' column")
    source.add_argument("--smiles", nargs="+", help="One or more SMILES strings")
    parser.add_argument("--output", type=Path, help="Output CSV (default: stdout)")
    args = parser.parse_args(argv)

    bundle = load_model_bundle(args.model)
    if args.input:
        rows, input_fields = _read_input(args.input)
        smiles = [row.get("smiles", "") for row in rows]
    else:
        rows, input_fields = [{"smiles": value} for value in args.smiles], ["smiles"]
        smiles = args.smiles
    predictions = predict_smiles(bundle, smiles)
    fields = input_fields + [name for name in OUTPUT_FIELDS if name not in input_fields]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    output = args.output.open("w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row, prediction in zip(rows, predictions):
            writer.writerow({**row, **prediction})
    finally:
        if args.output:
            output.close()


if __name__ == "__main__":
    main()
