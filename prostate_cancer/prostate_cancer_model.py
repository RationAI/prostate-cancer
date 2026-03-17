from copy import deepcopy

import torch
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import (
    AUROC,
    Accuracy,
    Metric,
    MetricCollection,
    NegativePredictiveValue,
    Precision,
    Recall,
    Specificity,
)

from prostate_cancer.typing import LabeledSampleBatch, UnlabeledSampleBatch


class ProstateCancerModel(LightningModule):
    def __init__(
        self,
        backbone: nn.Module | None = None,
        decode_head: nn.Module | None = None,
        full_model: nn.Module | None = None,
        lr: float = 1e-4,
    ) -> None:
        super().__init__()

        # enforce mutually exclusive configs
        if full_model is not None and (backbone is not None or decode_head is not None):
            raise ValueError(
                "Provide either `full_model` OR (`backbone` + `decode_head`), not both."
            )

        if full_model is None and decode_head is None:
            raise ValueError(
                "`decode_head` must be provided when using a backbone."
            )

        self.backbone = backbone
        self.decode_head = decode_head
        self.full_model = full_model
        self.lr = lr

        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

        metrics: dict[str, Metric | MetricCollection] = {
            "AUC": AUROC("binary"),
            "accuracy": Accuracy("binary"),
            "precision": Precision("binary"),
            "recall": Recall("binary"),
            "specificity": Specificity("binary"),
            "negative_predictive_value": NegativePredictiveValue("binary"),
        }

        self.train_metrics = MetricCollection(
            metrics=deepcopy(metrics),
            prefix="train/",
        )

        self.val_metrics = MetricCollection(
            metrics=deepcopy(metrics),
            prefix="validation/",
        )

        self.test_metrics = MetricCollection(
            metrics=deepcopy(metrics),
            prefix="test/",
        )

    def forward(self, x: Tensor) -> Tensor:
        # --- full model mode
        if self.full_model is not None:
            outputs = self.full_model(x)

            # HuggingFace models return objects
            if hasattr(outputs, "logits"):
                return outputs.logits

            return outputs

        # --- backbone is None in case of embeddings
        features = self.backbone(x) if self.backbone else x

        # --- if not full model, decode head must be present
        assert self.decode_head is not None, "Decode head must be present if not full model"
        logits = self.decode_head(features)
        return logits

    def _get_predictions(self, logits: Tensor) -> Tensor:
        return torch.sigmoid(logits)

    def training_step(self, batch: LabeledSampleBatch) -> Tensor:
        inputs, targets, _ = batch
        logits = self(inputs)
        predictions = self._get_predictions(logits)

        loss = self.criterion(logits, targets)
        self.log(
            "train/loss",
            loss,
            batch_size=len(inputs),
            on_step=True,
            prog_bar=True,
        )

        self.train_metrics.update(predictions, targets)
        self.log_dict(self.train_metrics, batch_size=len(inputs), on_epoch=True)

        return loss

    def validation_step(self, batch: LabeledSampleBatch) -> None:
        inputs, targets, _ = batch
        logits = self(inputs)
        predictions = self._get_predictions(logits)

        loss = self.criterion(logits, targets)
        self.log(
            "validation/loss",
            loss,
            batch_size=len(inputs),
            on_epoch=True,
            prog_bar=True,
        )

        self.val_metrics.update(predictions, targets)
        self.log_dict(self.val_metrics, batch_size=len(inputs), on_epoch=True)

    def test_step(
        self, batch: LabeledSampleBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        inputs, targets, _ = batch
        logits = self(inputs)
        predictions = self._get_predictions(logits)

        self.test_metrics.update(predictions, targets)

        return predictions

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute())

        self.test_metrics.reset()

    def predict_step(
        self, batch: UnlabeledSampleBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        inputs, _ = batch
        logits = self(inputs)
        return self._get_predictions(logits)

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), self.lr)
