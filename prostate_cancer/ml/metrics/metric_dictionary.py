# Copyright (c) The RationAI team.

import torch
import torchmetrics


class MetricDictionary(torch.nn.ModuleDict):
    """MetricDictionary.

    MetricDictionary a dynamic structure that automatically adds
    a new copy of predefined metric objects for each unique
    dataloader index.

    Additionally, a global metric "_global" is kept for each stage
    that calculates the metrics across all dataloaders.

    Stage can be any string but usually refers to the
    trainer stages: train, validate, test.
    """

    def __init__(self, default_metric_collection, pl_module, prefix, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super(torch.nn.Module, self).__setattr__(
            "default_metric_collection", default_metric_collection
        )
        super(torch.nn.Module, self).__setattr__("pl_module", pl_module)
        super(torch.nn.Module, self).__setattr__("prefix", prefix)

    def update(self, stage: str, y_pred, y_true, dataloader_idx=None) -> None:
        stage_key = self._key_from_stage(stage)
        dataloader_key = self._key_from_dataloader_idx(dataloader_idx)

        # Check if stage in MetricDictionary
        if stage_key not in self:
            self[stage_key] = torch.nn.ModuleDict()

        # Check if dataloader in MetricDictionary-stage
        if "_global" not in self[stage_key]:
            self[stage_key]["_global"] = self.default_metric_collection.clone(
                prefix=f"{stage}/"
            ).to(self.pl_module.device)
            self.register_metrics()
        if dataloader_idx is not None and dataloader_key not in self[stage_key]:
            self[stage_key][dataloader_key] = self.default_metric_collection.clone(
                prefix=f"{stage}/{dataloader_idx}/"
            ).to(self.pl_module.device)
            self.register_metrics()

        # Finally perform metric update
        self[stage_key]["_global"](y_pred, y_true)
        if dataloader_idx is not None:
            self[stage_key][dataloader_key](y_pred, y_true)

    def register_metrics(self):
        """Register metrics with the parent LightningModule.

        During the first call to the LightningModule.log() function, the
        LightningModule registers all metrics found within its submodule tree.
        Any additionally created metrics are ignored.

        To bypass this behaviour, we manually insert newly created metric
        collection into the LightningModule metric registry:
            id(metric_object) -> submodule.path.relative.to.lightningmodule

        As such it requires:
          1) access to the LightningModule instance
          2) name of the attribute under which the instance of MetricDictionary
             is saved under.

        Point 2) is the reason for the clunky register function in Classifier as
        getting the attribute name is nearly impossible for instance variables.
        """
        if self.pl_module._metric_attributes:
            for name, module in self.named_modules(prefix=self.prefix):
                if isinstance(module, torchmetrics.Metric):
                    self.pl_module._metric_attributes[id(module)] = name

    def compute(self, stage, dataloader_idx=None):
        stage_key = self._key_from_stage(stage)
        if dataloader_idx is None:
            return self[stage_key]["_global"].compute()

        return {
            key: self[stage_key][key].compute()
            for key in self[stage_key]
            if key != "_global"
        }

    def get(self, stage, dataloader_idx=None):
        stage_key = self._key_from_stage(stage)
        dataloader_key = self._key_from_dataloader_idx(dataloader_idx)
        return self[stage_key][dataloader_key]

    @staticmethod
    def _key_from_stage(stage: str) -> str:
        return f"{stage}_metrics"

    @staticmethod
    def _key_from_dataloader_idx(dataloader_idx: str | None) -> str:
        return dataloader_idx if dataloader_idx is not None else "_global"
