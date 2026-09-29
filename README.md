# MolML-Pipeline

This project does not propose a new bioactivity prediction method. It implements a reproducible ML engineering workflow using established molecular representations and benchmark datasets. V1 uses the **MoleculeNet BACE binary classification dataset** (1513 molecules: 822 class 0, 691 class 1). It makes no state-of-the-art claim.

## V1 method

RDKit parses SMILES and generates 2048-bit radius-2 Morgan fingerprints. A Logistic Regression and a 300-tree Random Forest use balanced class weights and fixed seed 42. The 80/10/10 random split is stratified. The primary benchmark uses whole, disjoint Bemis–Murcko scaffold groups, with a deterministic seeded search for partitions containing both labels and approximately matching target sizes and class balance. This split selection uses labels to ensure valid binary evaluation; it does not use model performance. The test set is never used for fitting or hyperparameter selection.

Random splitting can place structurally related molecules in different partitions. Scaffold splitting holds scaffold groups apart and provides a more demanding test of chemical generalization. The 0.5 probability threshold is fixed in advance for F1, Matthews correlation coefficient (MCC), balanced accuracy and the confusion matrix. ROC-AUC and PR-AUC (average precision) use positive-class probabilities. These are single seeded splits without uncertainty estimates; scaffold partition selection is heuristic and does not guarantee a representative population of future compounds. Different splits, label provenance and chemical series can change the measured performance. Predicted probabilities are model outputs and are **not experimentally validated biological activities or evidence of BACE1 inhibition**.

## Install and reproduce

Python 3.10+ and RDKit are required. A conda or mamba environment is recommended for RDKit.

```bash
conda create -n molml python=3.11 -y
conda activate molml
pip install -e ".[dev]"
mkdir -p data/raw
curl --fail --location https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv -o data/raw/bace.csv
python -m pytest -q
python -m molml.train --config configs/rf_scaffold.yaml --output results/v1/rf_scaffold
```

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

## Roadmap

V1.1 may add applicability-domain analysis and MLflow after the V1 benchmark. Later stages may add explainability and deployment interfaces. Interpretability scores or fingerprint bits must not be presented as experimentally established biological mechanisms.

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
