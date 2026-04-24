from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as pl
import mlflow
import pandas as pd
from rationai.masks.mask_builders import ScalarMaskBuilder
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle

from prostate_cancer.typing import LabeledTileSampleBatch, UnlabeledTileSampleBatch


if TYPE_CHECKING:
    from prostate_cancer.datamodule import TileDataModule


class HeatmapCallback(MultiloaderLifecycle):
    """Heatmap callback to save the heatmaps for positive class.

    Used in the test or predict stage to save the heatmaps for each slide.
    """

    def _on_dataloader_start(
        self,
        mode: str,
        trainer: pl.Trainer,
        dataloader_idx: int,
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        datamodule = cast("TileDataModule", trainer.datamodule)
        slide = cast("pd.Series", getattr(datamodule, mode).slides.iloc[dataloader_idx])

        # Create temporary directory
        tmp_dir = Path("tmp_dir")
        tmp_dir.mkdir(exist_ok=True)

        # Initialize the mask builder
        self.mask_builder: ScalarMaskBuilder = ScalarMaskBuilder(
            save_dir=tmp_dir,
            filename=Path(slide.path).stem,
            extent_x=slide.extent_x,
            extent_y=slide.extent_y,
            mpp_x=slide.mpp_x,
            mpp_y=slide.mpp_y,
            extent_tile=slide.tile_extent_x,
            stride=slide.stride_x,
        )

    def on_test_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        # Initialize the mask builder
        self._on_dataloader_start("test", trainer, dataloader_idx)

    def on_predict_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        # Initialize the mask builder
        self._on_dataloader_start("predict", trainer, dataloader_idx)

    def _on_batch_end(
        self,
        outputs: Any,
        batch: UnlabeledTileSampleBatch | LabeledTileSampleBatch,
    ) -> None:
        if len(batch) == 3:
            # Test step
            _, _, metadata = batch
        else:
            # Predict step
            _, metadata = batch

        # Update the mask builder with the outputs
        self.mask_builder.update(outputs.cpu(), metadata["x"], metadata["y"])

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

    def _on_dataloader_end(self) -> None:
        # Log the heatmap to MLFlow
        save_path = self.mask_builder.save()
        mlflow.log_artifact(str(save_path), artifact_path="heatmaps")

        # Delete the file
        save_path.unlink()

    def on_test_dataloader_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        self._on_dataloader_end()

    def on_predict_dataloader_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        self._on_dataloader_end()
