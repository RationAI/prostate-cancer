from typing import Any

from lightning import LightningModule, Trainer

from prostate_cancer.callbacks.tile_histograms_callback_base import (
    TileHistogramsCallbackBase,
)
from prostate_cancer.typing import LabeledTileSampleBatch


class TileHistogramsCallbackTile(TileHistogramsCallbackBase):
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: LabeledTileSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, y, _ = batch
        preds = outputs.detach().cpu().numpy().flatten()
        labels = y.detach().cpu().numpy().flatten()

        self.all_preds.append(preds)
        self.all_labels.append(labels)
