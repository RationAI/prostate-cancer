from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as pl
import mlflow
import pandas as pd
import torch
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle
from rationai.mlkit.metrics.aggregators import MeanPoolMaxAggregator

from prostate_cancer.typing import UnlabeledSampleBatch


if TYPE_CHECKING:
    from prostate_cancer.datamodule import DataModule


class KernelEstimationCallback(MultiloaderLifecycle):
    def __init__(self, kernel_sizes: list[int], extent_tile: int, stride: int) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.extent_tile = extent_tile
        self.stride = stride

    def on_predict_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        self.aggregators = [
            MeanPoolMaxAggregator(k, self.extent_tile, self.stride)
            for k in self.kernel_sizes
        ]
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
        for aggregator in self.aggregators:
            for i, (pred, target) in enumerate(zip(outputs, targets, strict=True)):
                aggregator.update(
                    preds=pred,
                    targets=target,
                    x=metadata["x"][i],
                    y=metadata["y"][i],
                )

    def on_predict_dataloader_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        # Compute the aggregated results for each kernel
        table: dict[str, Any] = {"slide_name": Path(self.slide.path).stem}

        for kernel_size, aggregator in zip(
            self.kernel_sizes, self.aggregators, strict=True
        ):
            pred, _ = aggregator.compute()
            table[f"pred_{kernel_size}"] = pred.item()

        if "carcinoma" in self.slide:
            table["target"] = self.slide["carcinoma"]

        mlflow.log_table(
            table,
            artifact_file="tables/aggregated_predictions.json",
        )
