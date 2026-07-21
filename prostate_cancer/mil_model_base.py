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


def binary_metrics(threshold: float) -> dict[str, Metric | MetricCollection]:
    return {
        "AUC": AUROC("binary"),
        "accuracy": Accuracy("binary", threshold=threshold),
        "precision": Precision("binary", threshold=threshold),
        "recall": Recall("binary", threshold=threshold),
        "specificity": Specificity("binary", threshold=threshold),
        "negative_predictive_value": NegativePredictiveValue(
            "binary", threshold=threshold
        ),
    }


class ProstateCancerMILBase(LightningModule):
    """Attention-MIL architecture shared by hybrid (SL+TL) and classic (SL-only) models.

    The bag encoder/attention/classifier and the forward pass are identical
    regardless of which labels supervise training. Both hybrid and classic
    models are evaluated at test time against SL *and* TL ground truth (TL
    metrics show how well the per-tile classifier localizes carcinoma even
    when, for the classic model, it was never directly supervised on TL
    labels) so `test_step` is shared here too. Subclasses only differ in
    which labels drive `training_step`/`validation_step`.
    """

    def __init__(
        self, foundation: str, lr: float, sl_threshold: float, tl_threshold: float
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

        # per-tile classifier, attention-pooled into the bag (SL) prediction
        self.classifier = nn.Linear(self.input_dim, 1)

        self.sl_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.lr = lr

        sl_metrics = binary_metrics(sl_threshold)
        self.train_metrics_sl = MetricCollection(
            deepcopy(sl_metrics), prefix="sl_train/"
        )
        self.val_metrics_sl = MetricCollection(
            deepcopy(sl_metrics), prefix="sl_validation/"
        )
        self.test_metrics_sl = MetricCollection(deepcopy(sl_metrics), prefix="sl_test/")

        # TL is only ever evaluated (not necessarily trained on) - see class docstring
        self.test_metrics_tl = MetricCollection(
            deepcopy(binary_metrics(tl_threshold)), prefix="tl_test/"
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

    def test_step(self, batch: LabeledBagOfTilesSampleBatch) -> MILModelOutput:  # type: ignore[override]
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
