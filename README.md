# MolML-Pipeline

A reproducible molecular machine-learning benchmark for binary bioactivity classification.

## Scope

MolML-Pipeline is an ML engineering project built around the established **BACE classification** benchmark from MoleculeNet. It does **not** propose a new bioactivity-prediction method or claim state-of-the-art performance. The goal is to implement a transparent, reproducible baseline using established cheminformatics methods.

## V1

The first version focuses on:

- validated SMILES input with RDKit
- Morgan fingerprints
- random and Bemis-Murcko scaffold splits
- Logistic Regression baseline
- Random Forest baseline
- ROC-AUC, PR-AUC, F1, MCC and balanced accuracy
- explicit split and class-balance reporting
- automated tests

The scaffold split is the primary evaluation setting because it tests generalization across molecular scaffolds more stringently than a random split.

## Scientific workflow

```text
MoleculeNet BACE
      |
      v
SMILES validation
      |
      v
RDKit Morgan fingerprints
      |
      v
Random split / Bemis-Murcko scaffold split
      |
      v
Logistic Regression / Random Forest
      |
      v
Classification metrics + confusion matrix
```

## Installation

A conda/mamba environment is recommended because RDKit is a compiled dependency.

```bash
conda create -n molml python=3.11 -y
conda activate molml
pip install -e ".[dev]"
```

## Data

Place the MoleculeNet BACE CSV at:

```text
data/raw/bace.csv
```

The loader intentionally does not silently substitute another dataset. It accepts the common MoleculeNet BACE schema and reports missing required columns clearly.

## Run

```bash
python -m molml.train --data data/raw/bace.csv --split scaffold --model random_forest
```

Use `--split random` for the secondary random-split comparison and `--model logistic_regression` for the linear baseline.

## Interpretation

Predictions produced by these models are benchmark model outputs, not experimentally validated biological activity. Later versions may add applicability-domain analysis, experiment tracking, explainability and deployment only after the V1 benchmark is validated.

## Roadmap

- V1: classical reproducible benchmark
- V1.1: applicability domain and experiment tracking
- V1.2: explainability
- V1.3: FastAPI, Docker and Streamlit

## License

MIT.
