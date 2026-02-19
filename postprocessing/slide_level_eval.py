import os
import tempfile

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torchmetrics import (
    Accuracy,
    NegativePredictiveValue,
    Precision,
    Recall,
    Specificity,
)
from torchmetrics.functional import confusion_matrix

from postprocessing.read_table import read_json_table


def evaluate(table: pd.DataFrame) -> tuple[dict[str, float], NDArray[np.floating]]:
    metrics = {
        "accuracy": Accuracy("binary"),
        "precision": Precision("binary"),
        "recall": Recall("binary"),
        "specificity": Specificity("binary"),
        "negative_predictive_value": NegativePredictiveValue("binary"),
    }

    target = torch.tensor(table["target"].astype(int).values)
    pred_binary = torch.tensor(table["pred_binary"].astype(int).values)

    results = {}
    for name, metric in metrics.items():
        value = metric(pred_binary, target)
        results[name] = value.item()

    cm = confusion_matrix(pred_binary, target, task="binary").cpu().numpy()
    return results, cm


def store_mispredictions(table: pd.DataFrame, path: str) -> None:
    mispreds = table[table["pred_binary"] != table["target"]]
    mispreds.to_csv(path, index=False)


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


@with_cli_args(["+postprocessing=slide_level_eval"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = read_json_table(config.preds_uri)
    df["pred_binary"] = df[config.pred_column] >= config.t
    results, cm = evaluate(df)
    logger.log_table(results, "slide_metrics.json")
    logger.log_metrics(results)

    with tempfile.TemporaryDirectory() as tmpdir:
        cm_path = os.path.join(tmpdir, "confusion_matrix.png")
        plot_and_save_confusion_matrix(cm, cm_path)
        logger.log_artifact(cm_path, artifact_path="plots")
        mp_path = os.path.join(tmpdir, "mispredictions.csv")
        store_mispredictions(df, mp_path)
        logger.log_artifact(mp_path, artifact_path="tables")


if __name__ == "__main__":
    main()
