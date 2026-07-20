from typing import TYPE_CHECKING, cast

import torch
from lightning import LightningModule, Trainer

from prostate_cancer.callbacks.curves_callback_base import CurvesCallbackBase
from prostate_cancer.typing import MILModelOutput, UnlabeledBagOfTilesSampleBatch


if TYPE_CHECKING:
    from prostate_cancer.datamodule import BagOfTilesDataModule


class CurvesCallbackSL(CurvesCallbackBase):
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

        targets = torch.tensor(
            [float(self._slide_targets[m["slide_id"]]) for m in metadata_batch]
        )

        self.preds.append(sl_outputs.detach().cpu())
        self.targets.append(targets)
