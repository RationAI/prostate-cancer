from abc import ABC
from collections.abc import Callable, Iterable
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
    SLLabeledBagOfTilesSample,
    SLLabeledBagOfTilesSampleBatch,
    UnlabeledBagOfTilesSample,
    UnlabeledBagOfTilesSampleBatch,
)


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


class BaseBagOfTilesDataModule(LightningDataModule, ABC):
    """Shared plumbing for bag-of-tiles (MIL) datamodules."""

    # set by subclasses to the collate_fn matching their train/val dataset's labels
    _collate_train_val: Callable[[list[Any]], Any]

    # test is always evaluated against SL+TL ground truth, regardless of subclass
    _collate_test = staticmethod(collate_fn_labeled)

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
            collate_fn=self._collate_train_val,
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
            collate_fn=self._collate_train_val,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Iterable[LabeledBagOfTilesSampleBatch]:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=self._collate_test,
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

    _collate_train_val = staticmethod(collate_fn_labeled)


class ClassicBagOfTilesDataModule(BaseBagOfTilesDataModule):
    """Datamodule for classic MIL: train/val only carry SL labels."""

    _collate_train_val = staticmethod(collate_fn_sl_labeled)
