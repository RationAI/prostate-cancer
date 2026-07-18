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
    LabeledBagOfTilesSampleBatch,
    MILModelOutput,
    UnlabeledBagOfTilesSampleBatch,
)


class ProstateCancerAttentionMIL(LightningModule):
    def __init__(
        self, foundation: str, lr: float, tl_threshold: float, sl_threshold: float
    ) -> None:
        super().__init__()
        match foundation:
            case "pgp":
                self.input_dim = 1536
            case "virchow2":
                self.input_dim = 2560
            case _:
                raise ValueError(f"Unknown foundation model: {foundation}")

        self.input_dim_sqrt = torch.tensor(self.input_dim).sqrt()

        # if we did not precompute the embeddings, we would obtain it from this module
        # (idendity replaced with foundation model)
        self.encoder = nn.Identity()

        # from a paper
        self.attention = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
        )

        # TL Classifier
        self.classifier = nn.Linear(self.input_dim, 1)

        self.sl_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.tl_criterion = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=torch.tensor([9.65])
        )  # handle padding
        self.lr = lr

        metrics: dict[str, dict[str, Metric | MetricCollection]] = {}

        # both SL and TL metrics
        for task_type, t in [("tl", tl_threshold), ("sl", sl_threshold)]:
            metrics[task_type] = {
                "AUC": AUROC("binary"),
                "accuracy": Accuracy("binary", threshold=t),
                "precision": Precision("binary", threshold=t),
                "recall": Recall("binary", threshold=t),
                "specificity": Specificity("binary", threshold=t),
                "negative_predictive_value": NegativePredictiveValue(
                    "binary", threshold=t
                ),
            }

        self.train_metrics_sl = MetricCollection(
            deepcopy(metrics["sl"]), prefix="sl_train/"
        )
        self.val_metrics_sl = MetricCollection(
            deepcopy(metrics["sl"]), prefix="sl_validation/"
        )
        self.test_metrics_sl = MetricCollection(
            deepcopy(metrics["sl"]), prefix="sl_test/"
        )

        self.train_metrics_tl = MetricCollection(
            deepcopy(metrics["tl"]), prefix="tl_train/"
        )
        self.val_metrics_tl = MetricCollection(
            deepcopy(metrics["tl"]), prefix="tl_validation/"
        )
        self.test_metrics_tl = MetricCollection(
            deepcopy(metrics["tl"]), prefix="tl_test/"
        )

    def forward(self, x: Tensor) -> MILModelOutput:
        # x has shape (batch_size, num_tiles_padded, embedding_dim)

        # Just identity
        x = self.encoder(x)  # (batch_size, num_tiles_padded, embedding_dim)

        # Do not attend to padded tiles (true for non-padded elements)
        mask = (
            (x.abs() > 1e-6).any(dim=-1, keepdim=True).float()
        )  # (batch_size, num_tiles_padded, 1)

        # TL weights (which tiles to attend to)
        raw_attn: Tensor = self.attention(x)  # (batch_size, num_tiles_padded, 1)
        raw_attn = raw_attn.masked_fill(
            ~mask.bool(), float("-inf")
        )  # (batch_size, num_tiles_padded, 1)

        # make it a distribution
        attention_weights = torch.softmax(
            raw_attn, dim=1
        )  # (batch_size, num_tiles_padded, 1)

        # TL predictions
        tl_preds_raw: Tensor = self.classifier(x)  # (batch_size, num_tiles_padded, 1)
        tl_preds_valid_raw = tl_preds_raw * mask

        # weight TL predictions with attention
        sl_pred_raw = torch.sum(
            attention_weights * tl_preds_valid_raw, dim=1
        )  # (batch_size, 1)

        return (
            sl_pred_raw.squeeze(-1),
            tl_preds_valid_raw.squeeze(-1),
            mask.squeeze(-1),
            attention_weights.squeeze(-1),
        )  # (batch_size,), (batch_size, num_tiles_padded), (batch_size, num_tiles_padded), (batch_size, num_tiles_padded)

    def training_step(self, batch: LabeledBagOfTilesSampleBatch) -> Tensor:
        # bag ~ all embeddings from a single slide
        bags, tl_labels, sl_labels, _ = batch

        sl_outputs, tl_outputs, mask, _ = self(bags)
        sl_loss = self.sl_criterion(sl_outputs, sl_labels)  # scalar
        tl_loss_all = self.tl_criterion(
            tl_outputs, tl_labels
        )  # (batch_size, num_tiles_padded)
        per_bag_tl_loss = (tl_loss_all * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        tl_loss = per_bag_tl_loss.mean()
        loss = sl_loss + tl_loss

        self.log("train/loss", loss, on_step=True, prog_bar=True, batch_size=len(bags))
        self.log(
            "train/sl_loss", sl_loss, on_step=True, prog_bar=True, batch_size=len(bags)
        )
        self.log(
            "train/tl_loss", tl_loss, on_step=True, prog_bar=True, batch_size=len(bags)
        )

        self.train_metrics_sl.update(sl_outputs, sl_labels)
        self.train_metrics_tl.update(tl_outputs[mask.bool()], tl_labels[mask.bool()])

        self.log_dict(
            self.train_metrics_sl, on_epoch=True, on_step=False, batch_size=len(bags)
        )
        self.log_dict(
            self.train_metrics_tl, on_epoch=True, on_step=False, batch_size=len(bags)
        )

        return loss

    def validation_step(self, batch: LabeledBagOfTilesSampleBatch) -> None:
        bags, tl_labels, sl_labels, _ = batch

        sl_outputs, tl_outputs, mask, _ = self(bags)
        sl_loss = self.sl_criterion(sl_outputs, sl_labels)
        tl_loss_all = self.tl_criterion(
            tl_outputs, tl_labels
        )  # (batch_size, num_tiles_padded)
        per_bag_tl_loss = (tl_loss_all * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        tl_loss = per_bag_tl_loss.mean()
        loss = sl_loss + tl_loss

        self.log("validation/loss", loss, prog_bar=True, batch_size=len(bags))
        self.log(
            "validation/sl_loss",
            sl_loss,
            on_step=True,
            prog_bar=True,
            batch_size=len(bags),
        )
        self.log(
            "validation/tl_loss",
            tl_loss,
            on_step=True,
            prog_bar=True,
            batch_size=len(bags),
        )

        self.val_metrics_sl.update(sl_outputs, sl_labels)
        self.val_metrics_tl.update(tl_outputs[mask.bool()], tl_labels[mask.bool()])

        self.log_dict(
            self.val_metrics_sl, on_epoch=True, on_step=False, batch_size=len(bags)
        )
        self.log_dict(
            self.val_metrics_tl, on_epoch=True, on_step=False, batch_size=len(bags)
        )

    def test_step(self, batch: LabeledBagOfTilesSampleBatch) -> MILModelOutput:
        bags, tl_labels, sl_labels, _ = batch

        sl_outputs, tl_outputs, mask, attention = self(bags)

        self.test_metrics_sl.update(sl_outputs, sl_labels)
        self.test_metrics_tl.update(tl_outputs[mask.bool()], tl_labels[mask.bool()])

        self.log_dict(
            self.test_metrics_sl, on_epoch=True, on_step=False, batch_size=len(bags)
        )
        self.log_dict(
            self.test_metrics_tl, on_epoch=True, on_step=False, batch_size=len(bags)
        )
        return sl_outputs.sigmoid(), tl_outputs.sigmoid(), mask, attention

    def predict_step(self, batch: UnlabeledBagOfTilesSampleBatch) -> MILModelOutput:
        sl_preds_raw, tl_preds_raw, mask, attention = self(batch[0])
        return sl_preds_raw.sigmoid(), tl_preds_raw.sigmoid(), mask, attention

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.lr)
