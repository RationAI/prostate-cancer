# Copyright (c) The RationAI team.

from abc import ABC
from typing import Any

import lightning
from lightning.pytorch.callbacks import Callback


class DataloaderAgnosticCallback(Callback, ABC):
    """An abstract callback class that has two new methods: on_test_dataloader_start and on_test_dataloader_end.

    The callback monitors the dataloader index in the on_test_batch_start method and calls the
    on_test_dataloader_start and on_test_dataloader_end methods when the dataloader index changes.
    """

    _current_dataloader_idx: int

    def __init__(self):
        super().__init__()
        self._current_dataloader_idx = -1

    @staticmethod
    def _extract_dataloader_metadata(batch: Any, outputs: dict) -> dict:
        """Extracts metadata from the batch and outputs.

        Metadata are assumed to be the same for all batches in a dataloader (slide)
        """
        excluded_keys = ["coord_x", "coord_y"]
        metadata = {
            key: val[0] for key, val in batch[2].items() if key not in excluded_keys
        }
        metadata["slide_channels"] = outputs["outputs"].shape[1]  # Assuming [N, C, ...]
        return metadata

    def on_test_batch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # monitor dataloader index
        if dataloader_idx != self._current_dataloader_idx:
            assert dataloader_idx > self._current_dataloader_idx, (
                "Dataloader indices must be increasing. "
                "The dataloaders are assumed to iterate in order."
            )
            if self._current_dataloader_idx >= 0:
                self.on_test_dataloader_end(
                    trainer=trainer,
                    pl_module=pl_module,
                    dataloader_idx=self._current_dataloader_idx,
                )

    def on_test_epoch_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        if self._current_dataloader_idx >= 0:
            self.on_test_dataloader_end(
                trainer=trainer,
                pl_module=pl_module,
                dataloader_idx=self._current_dataloader_idx,
            )

    def on_test_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: dict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx != self._current_dataloader_idx:
            metadata = self._extract_dataloader_metadata(batch=batch, outputs=outputs)

            self.on_test_dataloader_start(
                trainer=trainer,
                pl_module=pl_module,
                metadata=metadata,
                dataloader_idx=dataloader_idx,
            )

            self._current_dataloader_idx = dataloader_idx

    def on_test_dataloader_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        metadata: dict,
        dataloader_idx: int,
    ) -> None: ...

    def on_test_dataloader_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        dataloader_idx: int,
    ) -> None: ...
