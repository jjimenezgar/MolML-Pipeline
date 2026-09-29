"""Plain test-set diagnostic plots."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def save_test_plots(y_true, y_pred, y_prob, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, display in (
        ("roc", RocCurveDisplay.from_predictions(y_true, y_prob)),
        ("precision_recall", PrecisionRecallDisplay.from_predictions(y_true, y_prob)),
        ("confusion_matrix", ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1])),
    ):
        display.figure_.tight_layout()
        display.figure_.savefig(output_dir / f"{name}.png", dpi=150)
        plt.close(display.figure_)
