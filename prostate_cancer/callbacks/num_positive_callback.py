from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as pl
import pandas as pd
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle
from rationai.mlkit.lightning.loggers import MLFlowLogger

from prostate_cancer.datamodule.datasets.base import get_slide_name
from prostate_cancer.typing import UnlabeledTileSampleBatch


if TYPE_CHECKING:
    from prostate_cancer.datamodule import TileDataModule


class NumPositiveCallback(MultiloaderLifecycle):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

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

        datamodule = cast("DataModule", trainer.datamodule)
        slide = cast("pd.Series", datamodule.predict.slides.iloc[dataloader_idx])
        table = {"slide": get_slide_name(slide), "num_positive": self.num_positive}

        assert trainer.logger is not None
        assert isinstance(trainer.logger, MLFlowLogger)
        trainer.logger.log_table(table, artifact_file="num_positive_preds.json")
