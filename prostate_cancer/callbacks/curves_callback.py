from typing import Any

import mlflow
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from numpy.typing import NDArray
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from postprocessing.slide_level_curves import _plot_curve
from prostate_cancer.typing import LabeledTileSampleBatch


class CurvesCallback(Callback):
    def __init__(self, threshold: float, optimal_seek: bool = True) -> None:
        """This callback creates tile-level ROC curve and Precision-Recall curve and marks selected + optimized thresholds used for metric computation.

        Args:
            threshold (float): pathologist selected threshold
            optimal_seek (bool): whether we are looking for optimal thresholds or just want to plot the curves
        """
        super().__init__()
        self.optimal_seek = optimal_seek
        self.threshold = threshold
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: LabeledTileSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        targets = batch[1]
        self.preds.append(outputs.cpu())
        self.targets.append(targets.cpu())

    def _plot_roc(
        self, y_pred: NDArray[np.float32], y_true: NDArray[np.float32]
    ) -> None:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        # Find the point closest to the pre-selected threshold
        closest_idx = (np.abs(roc_thresholds - self.threshold)).argmin()
        pre_threshold = roc_thresholds[closest_idx]
        pre_fpr = fpr[closest_idx]
        pre_tpr = tpr[closest_idx]

        to_pinpoint = [(pre_fpr, pre_tpr)]
        labels = [f"Pre-selected Threshold = {pre_threshold:.2f}"]
        colors = ["red"]

        if self.optimal_seek:
            # J statistic to estimate threshold (maximize  TPR - FPR)
            j = tpr - fpr
            optimal_idx = j.argmax()
            j_threshold = roc_thresholds[optimal_idx]
            j_fpr = fpr[optimal_idx]
            j_tpr = tpr[optimal_idx]

            to_pinpoint.append((j_fpr, j_tpr))
            labels.append(f"J Threshold = {j_threshold:.2f}")
            colors.append("green")

        plot_path = "tile_roc.png"
        _plot_curve(
            fpr,
            tpr,
            f"AUC = {roc_auc:.3f}",
            to_pinpoint,
            labels,
            colors,
            "False Positive Rate",
            "True Positive Rate",
            "Receiver Operating Characteristic",
            plot_path,
            "lower right",
        )
        mlflow.log_artifact(plot_path, artifact_path="plots")

    def _plot_precision_recall(
        self, y_pred: NDArray[np.float32], y_true: NDArray[np.float32]
    ) -> None:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

        # Find the point closest to the pre-selected threshold
        closest_idx = (np.abs(thresholds - self.threshold)).argmin()
        pre_threshold = thresholds[closest_idx]
        to_pinpoint = [(recall[closest_idx], precision[closest_idx])]
        labels = [f"Pre-selected Threshold = {pre_threshold:.2f}"]
        colors = ["red"]

        if self.optimal_seek:
            # threshold maximizing F1 score
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1)
            best_threshold = thresholds[best_idx]

            to_pinpoint.append((recall[best_idx], precision[best_idx]))
            labels.append(f"F1 Threshold = {best_threshold:.2f}")
            colors.append("green")

        plot_path = "tile_precision_recall.png"
        _plot_curve(
            recall,
            precision,
            None,
            to_pinpoint,
            labels,
            colors,
            "Recall",
            "Precision",
            "Precision-Recall Curve",
            "tile_precision_recall.png",
            "lower left",
        )
        mlflow.log_artifact(plot_path, artifact_path="plots")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        y_pred = torch.cat(self.preds).numpy()
        y_true = torch.cat(self.targets).numpy()

        self._plot_roc(y_pred, y_true)
        self._plot_precision_recall(y_pred, y_true)

        self.preds.clear()
        self.targets.clear()
