import torch

from prostate_cancer.datamodule.bag_of_tiles_data_module import (
    BaseBagOfTilesDataModule,
)
from prostate_cancer.typing import (
    SLLabeledBagOfTilesSample,
    SLLabeledBagOfTilesSampleBatch,
)


class ClassicBagOfTilesDataModule(BaseBagOfTilesDataModule):
    """Datamodule for classic MIL: labeled samples only carry SL labels."""

    def _collate_labeled(
        self, batch: list[SLLabeledBagOfTilesSample]
    ) -> SLLabeledBagOfTilesSampleBatch:
        return collate_fn_sl_labeled(batch)


def collate_fn_sl_labeled(
    batch: list[SLLabeledBagOfTilesSample],
) -> SLLabeledBagOfTilesSampleBatch:
    inputs = []
    sl_labels = []
    metadatas = []
    for input, sl_label, metadata in batch:
        inputs.append(input)
        sl_labels.append(sl_label)
        metadatas.append(metadata)

    inputs_tensor = torch.stack(inputs)
    sl_labels_tensor = torch.stack(sl_labels)
    return inputs_tensor, sl_labels_tensor, metadatas
