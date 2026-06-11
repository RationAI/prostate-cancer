from lightning import LightningModule, Trainer

from prostate_cancer.callbacks.curves_callback_base import CurvesCallbackBase
from prostate_cancer.typing import LabeledSlideSampleBatch, MILModelOutput


class CurvesCallbackMIL(CurvesCallbackBase):
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: MILModelOutput, # type: ignore[override]
        batch: LabeledSlideSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        _, tl_outputs_raw, mask, _ = outputs
        tl_outputs_valid = tl_outputs_raw[mask.bool()]
        targets = batch[1]
        self.preds.append(tl_outputs_valid.flatten().cpu())
        self.targets.append(targets.flatten().cpu())
