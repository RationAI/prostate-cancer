from typing import Any

import lightning.pytorch as pl
from lightning import Callback
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.mlkit.metrics import NestedMetricCollection
from torchmetrics import (
    AUROC,
    Accuracy,
    NegativePredictiveValue,
    Precision,
    Recall,
    Specificity,
)

from prostate_cancer.typing import LabeledSampleBatch


class NestedMetricsCallback(Callback):
    """Calculates metrics using the `NestedMetricCollection` in the test stage."""

    def __init__(self, threshold: float) -> None:
        # In the test mode, log metrics for each slide
        self.nested_test_metrics = NestedMetricCollection(
            metrics={
                "AUC": AUROC("binary"),
                "accuracy": Accuracy("binary", threshold),
                "precision": Precision("binary", threshold),
                "recall": Recall("binary", threshold),
                "specificity": Specificity("binary", threshold),
                "negative_predictive_value": NegativePredictiveValue(
                    "binary", threshold
                ),
            }
        )

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: LabeledSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, targets, metadata = batch

        # Update slide-level metrics
        self.nested_test_metrics.update(outputs, targets, metadata["slide"])

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        assert isinstance(trainer.logger, MLFlowLogger)

        # Compute slide-level test metrics and log them
        trainer.logger.log_table(
            self.nested_test_metrics.compute(), "nested_metrics.json"
        )

        self.nested_test_metrics.reset()
