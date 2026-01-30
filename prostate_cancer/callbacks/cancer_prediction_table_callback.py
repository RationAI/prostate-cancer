from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as pl
import mlflow
import pandas as pd
import torch
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle

from prostate_cancer.typing import UnlabeledSampleBatch


if TYPE_CHECKING:
    from prostate_cancer.data import DataModule


class CancerPredictionTableCallback(MultiloaderLifecycle):
    """A callback to save predictions for tiles as a table."""

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def on_predict_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        datamodule = cast("DataModule", trainer.datamodule)
        self.slide = cast("pd.Series", datamodule.predict.slides.iloc[dataloader_idx])
        self.table: list[dict[str, Any]] = []

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

        for i, prediction in enumerate(outputs):
            self.table.append(
                {
                    "slide": Path(self.slide.path).stem,
                    "x": metadata["x"][i].item(),
                    "y": metadata["y"][i].item(),
                    "prediction": prediction.item(),
                    "binary_prediction": prediction.item() >= self.threshold,
                }
            )

    def on_predict_dataloader_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        table = pd.DataFrame(self.table)
        mlflow.log_table(table, "tables/cancer_prediction_table.json")
