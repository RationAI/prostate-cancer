from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as pl
import mlflow
import pandas as pd
import torch
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle

from prostate_cancer.typing import TilingSlideMetadata, UnlabeledTileSampleBatch


if TYPE_CHECKING:
    from prostate_cancer.datamodule import TileDataModule


class CarcinomaPredictionTableCallback(MultiloaderLifecycle):
    """A callback to save predictions for tiles as a table."""

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str | None = None,
    ) -> None:
        self.table: dict[str, Any] = {
            "slide": [],
            "x": [],
            "y": [],
            "prediction": [],
        }

    def on_predict_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

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

        for i, prediction in enumerate(outputs):
            self.table["slide"].append(Path(self.slide["path"]).stem)
            self.table["x"].append(metadata["x"][i].item())
            self.table["y"].append(metadata["y"][i].item())
            self.table["prediction"].append(prediction.item())

    def on_predict_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        df = pd.DataFrame(self.table)
        df.to_json("carcinoma_prediction_table.json", orient="split")
        mlflow.log_artifact(
            "carcinoma_prediction_table.json",
            artifact_path="tables",
        )
