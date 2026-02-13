from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as pl
import mlflow
import pandas as pd
import torch
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle
from rationai.mlkit.metrics.aggregators import Aggregator

from prostate_cancer.typing import UnlabeledSampleBatch


if TYPE_CHECKING:
    from prostate_cancer.datamodule import DataModule


class AggregatorCallback(MultiloaderLifecycle):
    def __init__(self, aggregator: Aggregator) -> None:
        super().__init__()
        self.aggregator_original = aggregator

    def on_predict_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")
        # aggregator cannot be reset, thus, its original state is copied for each slide
        self.aggregator = deepcopy(self.aggregator_original)
        datamodule = cast("DataModule", trainer.datamodule)
        self.slide = cast("pd.Series", datamodule.predict.slides.iloc[dataloader_idx])

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: UnlabeledSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, metadata = batch

        targets = torch.zeros_like(outputs)
        for i, (pred, target) in enumerate(zip(outputs, targets, strict=True)):
            self.aggregator.update(
                preds=pred,
                targets=target,
                x=metadata["x"][i],
                y=metadata["y"][i],
            )

    def on_predict_dataloader_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        # Compute the aggregated results
        pred, _ = self.aggregator.compute()
        table: dict[str, Any] = {
            "slide_name": Path(self.slide.path).stem,
            "prediction": pred.item(),
        }

        if "carcinoma" in self.slide:
            table["target"] = self.slide["carcinoma"]

        mlflow.log_table(
            table,
            artifact_file="tables/aggregated_predictions.json",
        )
