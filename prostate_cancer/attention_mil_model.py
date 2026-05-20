"""Original was created by Adam Kukučka in Ulcerative Colitis project."""

from copy import deepcopy

import torch
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    NegativePredictiveValue,
    Precision,
    Recall,
    Specificity,
)

from prostate_cancer.typing import (
    LabeledSlideSampleBatch,
    MILModelOutput,
    UnlabeledSlideSampleBatch,
)


class ProstateCancerAttentionMIL(LightningModule):
    def __init__(self, foundation: str, lr: float, tl_threshold: float) -> None:
        super().__init__()
        match foundation:
            case "pgp":
                input_dim = 1536
            case "virchow2":
                input_dim = 2560
            case _:
                raise ValueError(f"Unknown foundation model: {foundation}")

        self.encoder = nn.Identity()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
        )

        self.classifier = nn.Linear(input_dim, 1)
        self.sl_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.tl_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.lr = lr

        metrics: dict[str, Metric | MetricCollection] = {
            "AUC": AUROC("binary"),
            "accuracy": Accuracy("binary"),
            "precision": Precision("binary"),
            "recall": Recall("binary"),
            "specificity": Specificity("binary"),
            "negative_predictive_value": NegativePredictiveValue("binary"),
        }

        self.train_metrics_sl = MetricCollection(deepcopy(metrics), prefix="sl_train/")
        self.val_metrics_sl = MetricCollection(
            deepcopy(metrics), prefix="sl_validation/"
        )
        self.test_metrics_sl = MetricCollection(deepcopy(metrics), prefix="sl_test/")

        self.train_metrics_tl = MetricCollection(deepcopy(metrics), prefix="tl_train/")
        self.val_metrics_tl = MetricCollection(
            deepcopy(metrics), prefix="tl_validation/"
        )
        self.test_metrics_tl = MetricCollection(deepcopy(metrics), prefix="tl_test/")

    def forward(self, x: Tensor) -> MILModelOutput:
        # x has shape (batch_size, num_tiles_padded, embedding_dim)

        # Just identity
        x = self.encoder(x)  # (batch_size, num_tiles_padded, embedding_dim)

        # TL weights (which tiles to attend to)
        raw_attn = self.attention(x)  # (batch_size, num_tiles_padded, 1)

        # sigmoid is applied to avoid unimodal spiky distribution
        attention_weights = torch.softmax(
            raw_attn.sigmoid(), dim=1
        )  # (batch_size, num_tiles_padded, 1)

        # Do not attend to padded tiles
        mask = (
            (x.abs() > 1e-6).any(dim=-1, keepdim=True).float()
        )  # (batch_size, num_tiles_padded, 1)

        attention_weights = (
            attention_weights * mask
        )  # (batch_size, num_tiles_padded, 1)

        # proper distribution for the non-padded tiles
        attention_weights = attention_weights / attention_weights.sum(
            dim=1, keepdim=True
        )  # (batch_size, num_tiles_padded, 1)

        # TL predictions
        tl_preds_raw = self.classifier(x)  # (batch_size, num_tiles_padded, 1)
        tl_preds_valid_raw = tl_preds_raw * mask

        # weight TL predictions with attention
        sl_pred_raw = torch.sum(
            attention_weights * tl_preds_valid_raw, dim=1
        )  # (batch_size, 1)

        return (
            sl_pred_raw.squeeze(-1),
            tl_preds_valid_raw.squeeze(-1),
        )  # (batch_size,), (batch_size, num_tiles_padded)

    def training_step(self, batch: LabeledSlideSampleBatch) -> Tensor:
        # bag ~ all embeddings from a single slide
        bags, tl_labels, sl_labels, _ = batch

        sl_outputs, tl_outputs = self(bags)
        sl_loss = self.sl_criterion(sl_outputs, sl_labels)
        tl_loss = self.tl_criterion(tl_outputs, tl_labels)
        loss = sl_loss + tl_loss

        self.log("train/loss", loss, on_step=True, prog_bar=True, batch_size=len(bags))

        self.train_metrics_sl.update(sl_outputs, sl_labels)
        self.train_metrics_tl.update(tl_outputs, tl_labels)

        self.log_dict(
            self.train_metrics_sl, on_epoch=True, on_step=False, batch_size=len(bags)
        )
        self.log_dict(
            self.train_metrics_tl, on_epoch=True, on_step=False, batch_size=len(bags)
        )

        return loss

    def validation_step(self, batch: LabeledSlideSampleBatch) -> None:
        bags, tl_labels, sl_labels, _ = batch

        sl_outputs, tl_outputs = self(bags)
        sl_loss = self.sl_criterion(sl_outputs, sl_labels)
        tl_loss = self.tl_criterion(tl_outputs, tl_labels)
        loss = sl_loss + tl_loss

        self.log("validation/loss", loss, prog_bar=True, batch_size=len(bags))

        self.val_metrics_sl.update(sl_outputs, sl_labels)
        self.val_metrics_tl.update(tl_outputs, tl_labels)

        self.log_dict(
            self.val_metrics_sl, on_epoch=True, on_step=False, batch_size=len(bags)
        )
        self.log_dict(
            self.val_metrics_tl, on_epoch=True, on_step=False, batch_size=len(bags)
        )

    def test_step(self, batch: LabeledSlideSampleBatch) -> None:
        bags, tl_labels, sl_labels, _ = batch

        sl_outputs, tl_outputs = self(bags)

        self.test_metrics_sl.update(sl_outputs, sl_labels)
        self.test_metrics_tl.update(tl_outputs, tl_labels)

        self.log_dict(
            self.test_metrics_sl, on_epoch=True, on_step=False, batch_size=len(bags)
        )
        self.log_dict(
            self.test_metrics_tl, on_epoch=True, on_step=False, batch_size=len(bags)
        )

    def predict_step(self, batch: UnlabeledSlideSampleBatch) -> MILModelOutput:
        sl_preds_raw, tl_preds_raw = self(batch[0])
        return sl_preds_raw.sigmoid(), tl_preds_raw.sigmoid()

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.lr)
