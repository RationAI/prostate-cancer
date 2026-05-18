from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader


if TYPE_CHECKING:
    from prostate_cancer.datamodule.datasets import (
        LabeledSlideEmbeddingsDataset,
        UnlabeledSlideEmbeddingsDataset,
    )

from prostate_cancer.typing import (
    LabeledSlideSample,
    LabeledSlideSampleBatch,
    UnlabeledSlideSample,
    UnlabeledSlideSampleBatch,
)


class SlideDataModule(LightningDataModule):
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
                    "LabeledSlideEmbeddingsDataset",
                    instantiate(self.datasets["train"]),
                )
                self.val = cast(
                    "LabeledSlideEmbeddingsDataset",
                    instantiate(self.datasets["val"]),
                )
            case "val":
                self.val = cast(
                    "LabeledSlideEmbeddingsDataset",
                    instantiate(self.datasets["val"]),
                )
            case "test":
                self.test = cast(
                    "LabeledSlideEmbeddingsDataset",
                    instantiate(self.datasets["test"]),
                )
            case "predict":
                self.predict = cast(
                    "UnlabeledSlideEmbeddingsDataset",
                    instantiate(self.datasets["predict"]),
                )

    def train_dataloader(self) -> Iterable[LabeledSlideSampleBatch]:

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
            collate_fn=collate_fn_labeled,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> Iterable[LabeledSlideSampleBatch]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=collate_fn_labeled,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Iterable[LabeledSlideSampleBatch]:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=collate_fn_labeled,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self) -> Iterable[UnlabeledSlideSampleBatch]:
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            collate_fn=collate_fn_unlabeled,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


def collate_fn_labeled(batch: list[LabeledSlideSample]) -> LabeledSlideSampleBatch:
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
    batch: list[UnlabeledSlideSample],
) -> UnlabeledSlideSampleBatch:
    inputs = []
    metadatas = []
    for input, metadata in batch:
        inputs.append(input)
        metadatas.append(metadata)
    inputs_tensor = torch.stack(inputs)
    return inputs_tensor, metadatas
