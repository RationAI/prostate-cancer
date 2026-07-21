from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader


if TYPE_CHECKING:
    from prostate_cancer.datamodule.datasets import (
        BagOfEmbeddingsDataset,
        UnlabeledBagOfEmbeddingsDataset,
    )

from prostate_cancer.typing import (
    LabeledBagOfTilesSample,
    LabeledBagOfTilesSampleBatch,
    SLLabeledBagOfTilesSampleBatch,
    UnlabeledBagOfTilesSample,
    UnlabeledBagOfTilesSampleBatch,
)


class BaseBagOfTilesDataModule(LightningDataModule, ABC):
    """Shared plumbing for bag-of-tiles (MIL) datamodules.

    Subclasses only provide the collate function for labeled samples, since
    that is the only part that depends on which labels (SL only, or SL+TL)
    the underlying dataset produces.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        sampler: DictConfig | None = None,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = datasets
        self.sampler_partial = sampler

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self.train = cast(
                    "BagOfEmbeddingsDataset[Any]", instantiate(self.datasets["train"])
                )
                self.val = cast(
                    "BagOfEmbeddingsDataset[Any]", instantiate(self.datasets["val"])
                )
            case "val":
                self.val = cast(
                    "BagOfEmbeddingsDataset[Any]", instantiate(self.datasets["val"])
                )
            case "test":
                self.test = cast(
                    "BagOfEmbeddingsDataset[Any]", instantiate(self.datasets["test"])
                )
            case "predict":
                self.predict = cast(
                    "UnlabeledBagOfEmbeddingsDataset",
                    instantiate(self.datasets["predict"]),
                )

    @abstractmethod
    def _collate_labeled(self, batch: list[Any]) -> Any: ...

    def train_dataloader(
        self,
    ) -> Iterable[LabeledBagOfTilesSampleBatch | SLLabeledBagOfTilesSampleBatch]:

        if self.sampler_partial:
            sampler = instantiate(self.sampler_partial)(
                dataset=self.train, target_col="carcinoma"
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self.train,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=self._collate_labeled,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(
        self,
    ) -> Iterable[LabeledBagOfTilesSampleBatch | SLLabeledBagOfTilesSampleBatch]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=self._collate_labeled,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(
        self,
    ) -> Iterable[LabeledBagOfTilesSampleBatch | SLLabeledBagOfTilesSampleBatch]:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=self._collate_labeled,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self) -> Iterable[UnlabeledBagOfTilesSampleBatch]:
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            collate_fn=collate_fn_unlabeled,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


class BagOfTilesDataModule(BaseBagOfTilesDataModule):
    """Datamodule for hybrid MIL: labeled samples carry both SL and TL labels."""

    def _collate_labeled(
        self, batch: list[LabeledBagOfTilesSample]
    ) -> LabeledBagOfTilesSampleBatch:
        return collate_fn_labeled(batch)


def collate_fn_labeled(
    batch: list[LabeledBagOfTilesSample],
) -> LabeledBagOfTilesSampleBatch:
    inputs = []
    sl_labels = []
    tl_labels = []
    metadatas = []
    for input, sl_label, tl_label, metadata in batch:
        inputs.append(input)
        sl_labels.append(sl_label)
        tl_labels.append(tl_label)
        metadatas.append(metadata)

    inputs_tensor = torch.stack(inputs)
    sl_labels_tensor = torch.stack(sl_labels)
    tl_labels_tensor = torch.stack(tl_labels)
    return inputs_tensor, sl_labels_tensor, tl_labels_tensor, metadatas


def collate_fn_unlabeled(
    batch: list[UnlabeledBagOfTilesSample],
) -> UnlabeledBagOfTilesSampleBatch:
    inputs = []
    metadatas = []
    for input, metadata in batch:
        inputs.append(input)
        metadatas.append(metadata)
    inputs_tensor = torch.stack(inputs)
    return inputs_tensor, metadatas
