from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as pl
import mlflow
import pandas as pd
import torch
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle

from ml.typing import (
    LabeledTileSampleBatch,
    TilingSlideMetadata,
    UnlabeledTileSampleBatch,
)


if TYPE_CHECKING:
    from ml.datamodule import TileDataModule


class CarcinomaPredictionTableCallback(MultiloaderLifecycle):
    """A callback to save predictions for tiles as a table.

    Used in the test or predict stage to save the tile-level predictions.
    """

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

    def _on_dataloader_start(
        self, mode: str, trainer: pl.Trainer, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        datamodule = cast("TileDataModule", trainer.datamodule)
        self.slide = cast(
            "TilingSlideMetadata", getattr(datamodule, mode).slides[dataloader_idx]
        )

    def on_test_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        self._on_dataloader_start("test", trainer, dataloader_idx)

    def on_predict_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        self._on_dataloader_start("predict", trainer, dataloader_idx)

    def _on_batch_end(
        self,
        outputs: torch.Tensor,
        batch: UnlabeledTileSampleBatch | LabeledTileSampleBatch,
    ) -> None:
        if len(batch) == 3:
            # Test step
            _, _, metadata = batch
        else:
            # Predict step
            _, metadata = batch

        for i, prediction in enumerate(outputs):
            self.table["slide"].append(Path(self.slide["path"]).stem)
            self.table["x"].append(metadata["x"][i].item())
            self.table["y"].append(metadata["y"][i].item())
            self.table["prediction"].append(prediction.item())

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: LabeledTileSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end(outputs, batch)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: UnlabeledTileSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end(outputs, batch)

    def _save_table(self) -> None:
        df = pd.DataFrame(self.table)
        df.to_json("carcinoma_prediction_table.json", orient="split")
        mlflow.log_artifact(
            "carcinoma_prediction_table.json",
            artifact_path="tables",
        )

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        super().on_test_epoch_end(trainer, pl_module)
        self._save_table()

    def on_predict_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        super().on_predict_epoch_end(trainer, pl_module)
        self._save_table()
