# MolML-Pipeline

This project does not propose a new bioactivity prediction method. It implements a reproducible ML engineering workflow using established molecular representations and benchmark datasets. V1 uses the **MoleculeNet BACE binary classification dataset** (1513 molecules: 822 class 0, 691 class 1). It makes no state-of-the-art claim.

## V1 method

RDKit parses SMILES and generates 2048-bit radius-2 Morgan fingerprints. A Logistic Regression and a 300-tree Random Forest use balanced class weights and fixed seed 42. The 80/10/10 random split is stratified. The primary benchmark uses whole, disjoint Bemis–Murcko scaffold groups, with a deterministic seeded search for partitions containing both labels and approximately matching target sizes and class balance. This split selection uses labels to ensure valid binary evaluation; it does not use model performance. The test set is never used for fitting or hyperparameter selection.

Random splitting can place structurally related molecules in different partitions. Scaffold splitting holds scaffold groups apart and provides a more demanding test of chemical generalization. The 0.5 probability threshold is fixed in advance for F1, Matthews correlation coefficient (MCC), balanced accuracy and the confusion matrix. ROC-AUC and PR-AUC (average precision) use positive-class probabilities. These are single seeded splits without uncertainty estimates; scaffold partition selection is heuristic and does not guarantee a representative population of future compounds. Different splits, label provenance and chemical series can change the measured performance. Predicted probabilities are model outputs and are **not experimentally validated biological activities or evidence of BACE1 inhibition**.

## Install and reproduce

Python 3.10+ and RDKit are required. A conda or mamba environment is recommended for RDKit.

```bash
conda create -n molml python=3.12 -y
conda activate molml
pip install -c requirements-benchmark.txt -e ".[dev,explain]"
mkdir -p data/raw
curl --fail --location https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv -o data/raw/bace.csv
python -m pytest -q
python -m molml.train --config configs/rf_scaffold.yaml --output results/v1/rf_scaffold
```

The constraints file fixes the direct numerical and SHAP dependencies used for the published results. The original environment used Python 3.12.14. Other supported Python/dependency versions can pass the checks while producing slightly different metrics; the constraints are not a complete transitive lockfile.

For the full comparison, run the same command with each of `configs/lr_random.yaml`, `configs/lr_scaffold.yaml`, `configs/rf_random.yaml`, and `configs/rf_scaffold.yaml`, setting `--output results/v1/<config-stem>`. Each directory contains `metrics.json` and test-set ROC, precision–recall and confusion-matrix PNGs. Metrics JSON includes full configuration, dataset SHA-256, package versions, class counts, partition sizes, unique-scaffold counts and pairwise scaffold overlap. Do not treat the raw CSV as a repository artifact; download it from the official dataset source above. The original `--data`, `--split` and `--model` CLI arguments remain available for older commands.

## V1 benchmark

The following numbers were generated from the official CSV using the four checked-in configurations. They are results for these specific partitions, not optimized model selection. Class counts below use **inactive/active** order.

| Split | Partition | Molecules | Inactive / active | Unique scaffolds |
| --- | --- | ---: | ---: | ---: |
| Random | Train | 1210 | 657 / 553 | 575 |
| Random | Validation | 151 | 82 / 69 | 119 |
| Random | Test | 152 | 83 / 69 | 107 |
| Scaffold | Train | 1211 | 653 / 558 | 387 |
| Scaffold | Validation | 151 | 87 / 64 | 140 |
| Scaffold | Test | 151 | 82 / 69 | 144 |

Scaffold split overlap is zero for every pair of partitions. The random split has train–validation 64, train–test 65 and validation–test 26 shared unique scaffolds.

| Model | Split | Set | ROC-AUC | PR-AUC | F1 | MCC | Balanced accuracy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | Random | Validation | 0.866 | 0.833 | 0.757 | 0.548 | 0.774 |
| Logistic Regression | Random | Test | 0.906 | 0.858 | 0.774 | 0.588 | 0.794 |
| Logistic Regression | Scaffold | Validation | 0.889 | 0.857 | 0.810 | 0.673 | 0.835 |
| Logistic Regression | Scaffold | Test | 0.851 | 0.808 | 0.768 | 0.573 | 0.786 |
| Random Forest | Random | Validation | 0.863 | 0.853 | 0.748 | 0.534 | 0.767 |
| Random Forest | Random | Test | 0.905 | 0.876 | 0.791 | 0.616 | 0.808 |
| Random Forest | Scaffold | Validation | 0.898 | 0.857 | 0.772 | 0.606 | 0.802 |
| Random Forest | Scaffold | Test | 0.873 | 0.831 | 0.754 | 0.572 | 0.782 |

The random test results do not establish superior generalization to new chemical scaffolds. The single scaffold split is the primary V1 evaluation; no hyperparameter search or repeated test-set-driven model selection was performed. See [`results/v1`](results/v1) for unrounded machine-readable results, confusion matrices and plots.

## Project status and purpose

V1, V1.1, V1.2, V3, V4 and V5 are implemented. The project demonstrates a reproducible molecular ML workflow: **SMILES → validation → Morgan fingerprints → fixed random/scaffold partitions → baseline training → evaluation → similarity diagnostics → model attribution → local inference → HTTP API → Docker**. It can support discussion of candidate prioritization, but prospective utility on new compounds has not been validated.

| Stage | Delivered |
| --- | --- |
| V1 | Four fixed BACE experiments, leakage diagnostics, classification metrics and plots |
| V1.1 | Training-only similarity-domain diagnostics, scaffold novelty and optional local MLflow tracking |
| V1.2 | Checked SHAP attributions for eight fixed test molecules per scaffold model |
| V3 | Versioned local model bundle and batch predictions with per-molecule validation and training-domain diagnostics |
| V4 | Optional local FastAPI service that reuses V3 inference |
| V5 | Docker image and Compose setup for localhost API serving with a read-only model mount |

The next useful scientific step is repeated predeclared splits to measure partition variability. The container setup is intended for local development; an internet-facing deployment needs authentication and deployment-specific security controls. Interpretability scores or fingerprint bits must not be presented as experimentally established biological mechanisms.

## License

MIT.

## V1.1 — similarity domain and experiment tracking

V1.1 adds a **diagnostic** of how similar a query fingerprint is to its nearest training fingerprint, using Tanimoto similarity. The domain threshold is the fifth percentile of each training molecule's nearest *other* training molecule; validation and test data never determine it. Results also mark whether the Bemis–Murcko scaffold occurs in training. Similarity and scaffold novelty answer different questions: a new scaffold may still have high fingerprint similarity. The threshold is heuristic, not a calibrated uncertainty estimate or a guarantee of accuracy. No model or threshold was selected by test performance.

```bash
python -m molml.v11 --config configs/rf_scaffold.yaml --output results/v1.1/rf_scaffold
pip install -e ".[tracking]"
python -m molml.v11 --config configs/rf_scaffold.yaml --output results/v1.1/rf_scaffold --mlflow-uri sqlite:///mlflow.db
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

MLflow is opt-in and logs the fixed configuration, dataset hash, domain summary and CSV/JSON artifacts to a local SQLite store (ignored by Git). The four reproducible CSV/JSON results are in [`results/v1.1`](results/v1.1). For both models, the random test has 146/152 inside the heuristic similarity domain and 43/152 unseen scaffolds; the scaffold test has 126/151 inside the similarity domain and 151/151 unseen scaffolds. The split and fingerprint configuration determine these diagnostics, so their values coincide across models. Predicted probabilities in the CSV remain unvalidated model outputs.

## V1.2 — fingerprint attribution

V1.2 adds opt-in SHAP explanations for the two **unchanged scaffold-split V1 models**. It explains the first eight rows of the fixed test partition, selected by order before looking at predictions. Logistic Regression contributions add up to the class-1 **log-odds**; Random Forest contributions add up to the class-1 **predicted probability**. The command checks numerical additivity and fails if it exceeds 1e-4. Logistic Regression uses 64 training fingerprints sampled with the configured seed as its reference; Random Forest uses training tree-path statistics. Neither explainer uses validation or test labels to choose a baseline, tune the model or select samples.

```bash
pip install -e ".[explain]"
python -m molml.explain --config configs/lr_scaffold.yaml --output results/v1.2/lr_scaffold
python -m molml.explain --config configs/rf_scaffold.yaml --output results/v1.2/rf_scaffold
```

[`results/v1.2`](results/v1.2) contains summary metadata, a small set of local attributions and a bar chart of mean absolute attribution by fingerprint-bit index. **Hashed Morgan bits can collide and do not uniquely identify chemical fragments.** Attributions describe these fitted models on these examples; they are not causal effects, experimentally validated BACE1 inhibition, or evidence for a biological mechanism. The magnitudes from Logistic Regression and Random Forest have different units and should not be compared directly. Eight examples do not establish stable global importance.


## V3 — saved-model local inference

Train the reproducible scaffold-split baseline and save a local bundle containing the fitted model, fingerprint settings, training-only similarity reference, scaffold reference and dataset provenance:

```bash
python -m molml.train \\
  --config configs/rf_scaffold.yaml \\
  --output results/v3/rf_scaffold \\
  --save-model artifacts/v3/rf_scaffold.joblib
```

The `artifacts/` directory is ignored by Git. Make a CSV with a `smiles` column, then predict a batch:

```bash
python -m molml.predict \\
  --model artifacts/v3/rf_scaffold.joblib \\
  --input molecules.csv \\
  --output predictions.csv
```

For a few structures, use `--smiles "CCO" "c1ccccc1O"` instead of `--input`; omit `--output` to write CSV to the terminal. Each output row includes the class-1 model score, the fixed 0.5 threshold label, nearest-training Tanimoto similarity, the heuristic similarity-domain flag, scaffold novelty, and row-level input status. Invalid SMILES are reported per row without stopping the batch. These scores are uncalibrated model outputs, not experimentally validated activities. Similarity coverage is a heuristic, not an uncertainty or accuracy guarantee. Model bundles use joblib/pickle: load only bundles from sources you trust. This CLI is also the inference layer used by the V4 local API and V5 container.


## V4 — local HTTP API

Install the optional API dependencies (the core ML dependencies are installed with the package):

```bash
pip install -e ".[api]"
```

Start the service using the V3 model bundle created above. The default bind address is `127.0.0.1`, so it accepts requests only from your computer:

```bash
MOLML_MODEL_PATH=artifacts/v3/rf_scaffold.joblib molml-api
```

Check that the model loaded and open the interactive API documentation at `http://127.0.0.1:8000/docs`:

```bash
curl http://127.0.0.1:8000/health
curl --request POST \\
  --header 'Content-Type: application/json' \\
  --data '{"smiles":["CCO","not-a-smiles"]}' \\
  http://127.0.0.1:8000/predict
```

`POST /predict` accepts a JSON object with a `smiles` list and returns the V3 fields for each molecule. A batch can contain up to 128 SMILES; invalid entries return a row-level error. The model bundle loads once when the service starts. Keep this API on localhost: it has no authentication and is intended for local development, not public deployment.


## V5 — Docker

Build the V3 model bundle first using the command above. Docker Compose builds the API image, mounts the model read-only at runtime, and publishes port 8000 only on localhost. The model file is excluded from the image and must remain a trusted bundle.

On Linux, run Compose with your user and group IDs so the container can read the locally generated model file:

```bash
MOLML_UID=$(id -u) MOLML_GID=$(id -g) docker compose up --build
```

On Docker Desktop, the default Compose IDs usually work:

```bash
docker compose up --build
```

Check `http://127.0.0.1:8000/health` or open `http://127.0.0.1:8000/docs`. Send a prediction request with:

```bash
curl --request POST \\
  --header 'Content-Type: application/json' \\
  --data '{"smiles":["CCO","not-a-smiles"]}' \\
  http://127.0.0.1:8000/predict
```

Set `MOLML_MODEL_FILE=/path/to/model.joblib` to mount a different V3 bundle. Stop the service with `docker compose down`. Requires Docker Engine and Docker Compose v2. The image does not contain model data, and this unauthenticated API is bound to localhost for development only.
