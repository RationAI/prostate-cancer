from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, cast

import lightning.pytorch as pl
import mlflow
import pandas as pd
import torch
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle
from rationai.mlkit.metrics.aggregators import Aggregator

from prostate_cancer.typing import TilingSlideMetadata, UnlabeledTileSampleBatch


if TYPE_CHECKING:
    from prostate_cancer.datamodule import TileDataModule


class AggregatorCallback(MultiloaderLifecycle):
    def __init__(self, aggregator: Aggregator) -> None:
        super().__init__()
        self.aggregator_original = aggregator

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str | None = None,
    ) -> None:
        self.table = {
            "slide_name": [],
            "prediction": [],
            "target": [],
        }

    def on_predict_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")
        # aggregator cannot be reset, thus, its original state is copied for each slide
        self.aggregator = deepcopy(self.aggregator_original)
        datamodule = cast("TileDataModule", trainer.datamodule)
        self.slide = cast(
            "TilingSlideMetadata", datamodule.predict.slides[dataloader_idx]
        )

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: UnlabeledTileSampleBatch,
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

        self.table["slide_name"].append(Path(self.slide["path"]).stem)
        self.table["prediction"].append(pred.item())
        if "carcinoma" in self.slide:
            self.table["target"].append(self.slide["carcinoma"])

    def on_predict_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        df = pd.DataFrame(self.table)
        df.to_json("aggregated_predictions.json", orient="split")
        mlflow.log_artifact(
            "aggregated_predictions.json",
            artifact_path="tables",
        )
