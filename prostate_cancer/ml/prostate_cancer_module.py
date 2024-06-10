# Copyright (c) The RationAI team.

import functools
import logging
from collections import OrderedDict

import lightning
import torch
import torchmetrics

from prostate_cancer.ml.metrics import MetricDictionary


logger = logging.getLogger("prostate_cancer_module")


class ProstateCancerModule(lightning.pytorch.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module | dict[str, torch.nn.Module],
        output_activation: torch.nn.Module | None = None,
        loss: torch.nn.Module | None = None,
        optimizer: functools.partial | None = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        metrics: dict[str, torchmetrics.Metric] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(net, dict):
            net = torch.nn.Sequential(OrderedDict(net))
        self.model = net
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.output_activation = output_activation

        self.register_metric_dict("metrics", metrics)

    def forward(self, x):
        x = self.model(x)
        x = self.output_activation(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self(x)

        # Update and log loss
        loss = self.loss(y_pred, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True)

        # Update and log metrics
        self.metrics.update("train", y_pred, y)
        self.log_dict(self.metrics.get("train"), on_step=True)

        return {
            "loss": loss,
            "metrics": self.metrics.compute("train"),
            "outputs": y_pred,
        }

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self(x)

        # Update and log loss
        loss = self.loss(y_pred, y)
        self.log("valid/loss", loss)

        # Update and log metrics
        self.metrics.update("valid", y_pred, y)
        self.log_dict(self.metrics.get("valid"), add_dataloader_idx=False)

        return {
            "loss": loss,
            "metrics": self.metrics.compute("valid"),
            "outputs": y_pred,
        }

    def on_validation_epoch_end(self):
        self.log_dict(self.metrics.get("valid"))

    @torch.inference_mode()
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        x, y, metadata = batch
        y_pred = self(x)

        # Update and log metrics
        if "slide_name" in metadata:
            slide_name = metadata["slide_name"][0]
        else:
            slide_name = dataloader_idx
        self.metrics.update("test", y_pred, y, slide_name)
        self.log_dict(self.metrics.get("test", slide_name), add_dataloader_idx=False)

        return {"metrics": self.metrics.compute("test", slide_name), "outputs": y_pred}

    def on_test_epoch_end(self):
        self.log_dict(self.metrics.get("test"))

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y, _ = batch
        y_pred = self(x)
        return {"outputs": y_pred}

    def configure_optimizers(self):
        if self.optimizer is None:
            return None

        optimizer = self.optimizer(self.model.parameters())
        ret = {"optimizer": optimizer}

        if self.lr_scheduler:
            ret["lr_scheduler"] = {
                "scheduler": self.lr_scheduler["scheduler"](optimizer)
            }

            monitor = self.lr_scheduler.get("monitor")
            if monitor:
                ret["lr_scheduler"]["monitor"] = monitor

        return ret

    def register_metric_dict(self, parameter_name: str, metrics: dict | None) -> None:
        metric_collection = torchmetrics.MetricCollection(metrics or {})
        metric_module = MetricDictionary(metric_collection, self, parameter_name)
        self.add_module(parameter_name, metric_module)
