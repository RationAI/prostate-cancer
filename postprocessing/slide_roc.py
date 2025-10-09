import json
from pathlib import Path
from typing import cast

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from lightning.pytorch.loggers import Logger
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def _plot_curve(
    xs: NDArray[np.float32],
    ys: NDArray[np.float32],
    plot_label: str | None,
    to_pinpoint: list[tuple[np.float32, np.float32]],
    point_labels: list[str],
    point_colors: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    plot_path_str: str,
    loc: str,
) -> None:
    plt.figure()
    plt.plot(xs, ys, label=plot_label)

    for i in range(len(to_pinpoint)):
        x, y = to_pinpoint[i]
        plt.scatter(
            x,
            y,
            color=point_colors[i],
            label=point_labels[i],
            zorder=5,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=loc)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(plot_path_str)
    plt.close()


def perform_roc(table: pd.DataFrame) -> tuple[str, np.float32]:
    fpr, tpr, thresholds = roc_curve(table["target"], table["prediction"])

    roc_auc = auc(fpr, tpr)
    idx = np.where(tpr == 1)[0]
    best_idx = idx[np.argmin(fpr[idx])]
    best_threshold = thresholds[best_idx]
    plot_path = "slide_roc.png"
    _plot_curve(
        fpr,
        tpr,
        f"AUC = {roc_auc:.3f}",
        [(fpr[best_idx], tpr[best_idx])],
        [
            f"TPR Threshold = {best_threshold:.2f}",
        ],
        ["red"],
        "False Positive Rate",
        "True Positive Rate",
        "Receiver Operating Characteristic",
        plot_path,
        "lower right",
    )

    return plot_path, best_threshold


def perform_pr(table: pd.DataFrame) -> tuple[str, np.float32]:
    precision, recall, thresholds = precision_recall_curve(
        table["target"], table["prediction"]
    )

    # threshold maximizing F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1)
    best_threshold = thresholds[best_idx]

    plot_path = "slide_precision_recall.png"
    _plot_curve(
        recall,
        precision,
        None,
        [
            (recall[best_idx], precision[best_idx]),
        ],
        [
            f"F1 Threshold = {best_threshold:.2f}",
        ],
        ["red"],
        "Recall",
        "Precision",
        "Precision-Recall Curve",
        plot_path,
        "lower left",
    )

    return plot_path, best_threshold


@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    assert logger is not None, "Need logger"
    logger = cast("MLFlowLogger", logger)

    table_path = mlflow.artifacts.download_artifacts(config.preds_uri)
    with open(table_path) as file:
        json_data = json.load(file)

    df = pd.DataFrame(json_data["data"], columns=json_data["columns"])
    roc_path, roc_t = perform_roc(df)

    logger.experiment.log_artifact(
        run_id=logger.run_id,
        local_path=roc_path,
        artifact_path="plots",
    )
    logger.experiment.log_param(logger.run_id, "roc_threshold", roc_t)

    pr_path, pr_t = perform_pr(df)

    logger.experiment.log_artifact(
        run_id=logger.run_id,
        local_path=pr_path,
        artifact_path="plots",
    )
    logger.experiment.log_param(logger.run_id, "pr_threshold", pr_t)

    for plt_path in [roc_path, pr_path]:
        Path(plt_path).unlink()


if __name__ == "__main__":
    main()
