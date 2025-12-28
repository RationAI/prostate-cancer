import os
import tempfile

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torchmetrics import (
    AUROC,
    Accuracy,
    NegativePredictiveValue,
    Precision,
    Recall,
    Specificity,
)
from torchmetrics.functional import confusion_matrix

from postprocessing.read_table import read_json_table


def evaluate(
    table: pd.DataFrame, t: float, pred_column: str
) -> tuple[dict[str, float], NDArray[np.floating]]:
    metrics = {
        "AUC": AUROC("binary"),
        "accuracy": Accuracy("binary"),
        "precision": Precision("binary"),
        "recall": Recall("binary"),
        "specificity": Specificity("binary"),
        "negative_predictive_value": NegativePredictiveValue("binary"),
    }

    target = torch.tensor(table["target"].astype(int).values)
    pred_prob = torch.tensor(table[pred_column].astype(float).values)
    pred_label = (pred_prob >= t).int()

    results = {}
    for name, metric in metrics.items():
        if name == "AUC":
            value = metric(pred_prob, target)
        else:
            value = metric(pred_label, target)
        results[name] = value.item()

    cm = confusion_matrix(pred_label, target, task="binary").cpu().numpy()
    return results, cm


def plot_and_save_confusion_matrix(cm: NDArray[np.floating], path: str) -> None:
    _, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


@hydra.main(
    config_path="../configs",
    config_name="postprocessing/slide_level_eval",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = read_json_table(config.preds_uri)
    results, cm = evaluate(df, config.t, config.pred_column)
    logger.log_table(results, "slide_metrics.json")
    logger.log_metrics(results)

    with tempfile.TemporaryDirectory() as tmpdir:
        cm_path = os.path.join(tmpdir, "confusion_matrix.png")
        plot_and_save_confusion_matrix(cm, cm_path)
        logger.log_artifact(cm_path, artifact_path="plots")


if __name__ == "__main__":
    main()
