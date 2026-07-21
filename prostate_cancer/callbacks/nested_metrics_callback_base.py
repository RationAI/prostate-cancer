from abc import ABC

import mlflow
import pandas as pd
from lightning import Callback, LightningModule, Trainer
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


class NestedMetricsCallbackBase(Callback, ABC):
    """Calculates metrics using the `NestedMetricCollection` in the test stage, grouped per slide."""

    def __init__(self, threshold: float) -> None:
        super().__init__()
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

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        assert isinstance(trainer.logger, MLFlowLogger)

        metrics = self.nested_test_metrics.compute()
        pd.DataFrame(metrics).to_json("nested_metrics.json", orient="split")
        mlflow.log_artifact("nested_metrics.json")
        self.nested_test_metrics.reset()
