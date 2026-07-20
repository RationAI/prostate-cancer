from typing import TYPE_CHECKING, cast

import numpy as np
from lightning import LightningModule, Trainer

from prostate_cancer.callbacks.tile_histograms_callback_base import (
    TileHistogramsCallbackBase,
)
from prostate_cancer.typing import MILModelOutput, UnlabeledBagOfTilesSampleBatch


if TYPE_CHECKING:
    from prostate_cancer.datamodule import BagOfTilesDataModule


class SlideHistogramsCallbackMIL(TileHistogramsCallbackBase):
    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: str | None = None
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        datamodule = cast("BagOfTilesDataModule", trainer.datamodule)
        slides = datamodule.predict.slides
        self._slide_targets = dict(zip(slides["id"], slides["carcinoma"], strict=True))

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: MILModelOutput,
        batch: UnlabeledBagOfTilesSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        sl_outputs, _, _, _ = outputs
        _, metadata_batch = batch

        preds = sl_outputs.detach().cpu().numpy().flatten()
        labels = np.array(
            [float(self._slide_targets[m["slide_id"]]) for m in metadata_batch],
            dtype=np.float32,
        )

        self.all_preds.append(preds)
        self.all_labels.append(labels)
