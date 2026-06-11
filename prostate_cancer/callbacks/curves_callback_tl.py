from typing import Any

from lightning import LightningModule, Trainer

from prostate_cancer.callbacks.curves_callback_base import CurvesCallbackBase
from prostate_cancer.typing import LabeledTileSampleBatch


class CurvesCallbackTile(CurvesCallbackBase):
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: LabeledTileSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        targets = batch[1]
        self.preds.append(outputs.cpu())
        self.targets.append(targets.cpu())
