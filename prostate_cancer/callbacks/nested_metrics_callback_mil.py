from lightning import LightningModule, Trainer

from prostate_cancer.callbacks.nested_metrics_callback_base import (
    NestedMetricsCallbackBase,
)
from prostate_cancer.typing import LabeledBagOfTilesSampleBatch, MILModelOutput


class NestedMetricsCallbackMIL(NestedMetricsCallbackBase):
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: MILModelOutput,  # type: ignore[override]
        batch: LabeledBagOfTilesSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, tl_targets, _, metadata = batch
        _, tl_outputs_raw, mask, _ = outputs

        mask_bool = mask.bool()
        tl_outputs_valid = tl_outputs_raw[mask_bool]
        targets_valid = tl_targets[mask_bool]

        keys = [
            metadata[i]["slide_name"]
            for i in range(mask_bool.shape[0])
            for j in range(mask_bool.shape[1])
            if mask_bool[i, j]
        ]

        # Update slide-level metrics
        self.nested_test_metrics.update(
            tl_outputs_valid.cpu(), targets_valid.cpu(), keys
        )
