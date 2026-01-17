from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from postprocessing.read_table import read_json_table


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


def perform_roc(
    table: pd.DataFrame, pred_column: str
) -> tuple[str, np.floating, np.floating]:
    fpr, tpr, thresholds = roc_curve(table["target"], table[pred_column])

    roc_auc = auc(fpr, tpr)

    # TPR Threshold
    idx = np.where(tpr == 1)[0]
    tpr_idx = idx[np.argmin(fpr[idx])]
    tpr_threshold = thresholds[tpr_idx]

    # J Threshold
    j_scores = tpr - fpr
    j_idx = np.argmax(j_scores)
    j_threshold = thresholds[j_idx]

    plot_path = "slide_roc.png"
    _plot_curve(
        fpr,
        tpr,
        f"AUC = {roc_auc:.3f}",
        [(fpr[tpr_idx], tpr[tpr_idx]), (fpr[j_idx], tpr[j_idx])],
        [f"TPR Threshold = {tpr_threshold:.3f}", f"J Threshold = {j_threshold:.3f}"],
        ["red", "green"],
        "False Positive Rate",
        "True Positive Rate",
        "Receiver Operating Characteristic",
        plot_path,
        "lower right",
    )

    return plot_path, tpr_threshold, j_threshold


def perform_pr(table: pd.DataFrame, pred_column: str) -> tuple[str, np.float32]:
    precision, recall, thresholds = precision_recall_curve(
        table["target"], table[pred_column]
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
            f"F1 Threshold = {best_threshold:.3f}",
        ],
        ["red"],
        "Recall",
        "Precision",
        "Precision-Recall Curve",
        plot_path,
        "lower left",
    )

    return plot_path, best_threshold


@hydra.main(
    config_path="../configs",
    config_name="postprocessing/slide_level_curves",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = read_json_table(config.preds_uri)
    roc_path, roc_t, j_t = perform_roc(df, config.pred_column)
    pr_path, pr_t = perform_pr(df, config.pred_column)
    logger.log_artifact(local_path=roc_path, artifact_path="plots")
    logger.log_artifact(local_path=pr_path, artifact_path="plots")
    logger.log_hyperparams(
        {"roc_threshold": roc_t, "pr_threshold": pr_t, "j_threshold": j_t}
    )

    for plt_path in [roc_path, pr_path]:
        Path(plt_path).unlink()


if __name__ == "__main__":
    main()
