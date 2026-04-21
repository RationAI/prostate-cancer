from copy import deepcopy

import torch
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection, Metric
from torchmetrics.classification import (
    AUROC, Recall, Precision, Accuracy, NegativePredictiveValue, Specificity
)

from prostate_cancer.typing import LabeledSlideSampleBatch, UnlabeledSlideSampleBatch


class ProstateCancerAttentionMIL(LightningModule):
    def __init__(self, foundation: str, lr: float) -> None:
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
        self.criterion = nn.BCELoss()
        self.lr = lr

        metrics: dict[str, Metric | MetricCollection] = {
            "AUC": AUROC("binary"),
            "accuracy": Accuracy("binary"),
            "precision": Precision("binary"),
            "recall": Recall("binary"),
            "specificity": Specificity("binary"),
            "negative_predictive_value": NegativePredictiveValue("binary")
        }

        self.train_metrics = MetricCollection(deepcopy(metrics), prefix="train/")
        self.val_metrics = MetricCollection(deepcopy(metrics), prefix="validation/")
        self.test_metrics = MetricCollection(deepcopy(metrics), prefix="test/")

    def forward(self, x: Tensor) -> Tensor:
        # x has shape (batch_size, num_tiles_padded, embedding_dim)
        x = self.encoder(x)
        attn = self.attention(x)
        attention_weights = torch.softmax(attn.sigmoid(), dim=0)
        mask = (x.abs() > 1e-6).any(dim=-1, keepdim=True).float()
        attention_weights = attention_weights * mask
        attention_weights = attention_weights / attention_weights.sum(
            dim=1, keepdim=True
        )
        x = self.classifier(x)
        x = torch.sum(attention_weights * x, dim=1)
        x = x.sigmoid()

        return x.squeeze(-1)

    def training_step(self, batch: LabeledSlideSampleBatch) -> Tensor:
        bags, labels, _ = batch

        outputs = self(bags)
        loss = self.criterion(outputs, labels)
        self.log("train/loss", loss, on_step=True, prog_bar=True, batch_size=len(bags))

        self.train_metrics.update(outputs, labels)
        self.log_dict(
            self.train_metrics, on_epoch=True, on_step=False, batch_size=len(bags)
        )

        return loss

    def validation_step(self, batch: LabeledSlideSampleBatch) -> None:
        bags, labels, _ = batch

        outputs = self(bags)
        loss = self.criterion(outputs, labels)
        self.log("validation/loss", loss, prog_bar=True, batch_size=len(bags))

        self.val_metrics.update(outputs, labels)
        self.log_dict(
            self.val_metrics, on_epoch=True, on_step=False, batch_size=len(bags)
        )

    def test_step(self, batch: LabeledSlideSampleBatch) -> None:
        bags, labels, _ = batch

        outputs = self(bags)

        self.test_metrics.update(outputs, labels)
        self.log_dict(
            self.test_metrics, on_epoch=True, on_step=False, batch_size=len(bags)
        )

        return outputs

    def predict_step(self, batch: UnlabeledSlideSampleBatch) -> Tensor:
        return self(batch[0])

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.lr)
