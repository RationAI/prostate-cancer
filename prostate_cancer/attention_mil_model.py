"""Original was created by Adam Kukučka in Ulcerative Colitis project."""

from copy import deepcopy

import torch
from torch import Tensor, nn
from torchmetrics import MetricCollection

from prostate_cancer.mil_model_base import ProstateCancerMILBase, binary_metrics
from prostate_cancer.typing import LabeledBagOfTilesSampleBatch


class ProstateCancerAttentionMIL(ProstateCancerMILBase):
    """Hybrid MIL: trained on both slide-level (SL) and tile-level (TL) labels."""

    def __init__(
        self, foundation: str, lr: float, tl_threshold: float, sl_threshold: float
    ) -> None:
        super().__init__(
            foundation=foundation,
            lr=lr,
            sl_threshold=sl_threshold,
            tl_threshold=tl_threshold,
        )

        self.tl_criterion = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=torch.tensor([9.65])
        )  # handle padding

        tl_metrics = binary_metrics(tl_threshold)
        self.train_metrics_tl = MetricCollection(
            deepcopy(tl_metrics), prefix="tl_train/"
        )
        self.val_metrics_tl = MetricCollection(
            deepcopy(tl_metrics), prefix="tl_validation/"
        )

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
