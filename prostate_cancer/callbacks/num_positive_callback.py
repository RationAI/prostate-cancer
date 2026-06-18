from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as pl
import mlflow
import pandas as pd
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle

from prostate_cancer.datamodule.datasets.base import get_slide_name
from prostate_cancer.typing import TilingSlideMetadata, UnlabeledTileSampleBatch


if TYPE_CHECKING:
    from prostate_cancer.datamodule import TileDataModule


class NumPositiveCallback(MultiloaderLifecycle):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold
        self.table: dict[str, Any] = {
            "slide": [],
            "num_positive": [],
        }

    def on_predict_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        self.num_positive: int = 0

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: UnlabeledTileSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.num_positive += (outputs > self.threshold).sum().item()

    def on_predict_dataloader_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        datamodule = cast("TileDataModule", trainer.datamodule)
        slide = cast("TilingSlideMetadata", datamodule.predict.slides[dataloader_idx])
        self.table["slide"].append(get_slide_name(slide))
        self.table["num_positive"].append(self.num_positive)

    def on_predict_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        df = pd.DataFrame(self.table)
        df.to_json("num_positive_preds.json", orient="split")
        mlflow.log_artifact(
            "num_positive_preds.json",
            artifact_path="tables",
        )
