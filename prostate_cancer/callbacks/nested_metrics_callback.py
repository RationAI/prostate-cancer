from typing import Any

from lightning import LightningModule, Trainer

from prostate_cancer.callbacks.nested_metrics_callback_base import (
    NestedMetricsCallbackBase,
)
from prostate_cancer.typing import LabeledTileSampleBatch


class NestedMetricsCallback(NestedMetricsCallbackBase):
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: LabeledTileSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, targets, metadata = batch

        # Update slide-level metrics
        self.nested_test_metrics.update(outputs, targets, metadata["slide"])
