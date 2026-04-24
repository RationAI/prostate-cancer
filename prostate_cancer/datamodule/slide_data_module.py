from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

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
    LabeledSlideSampleBatch,
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
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> Iterable[LabeledSlideSampleBatch]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Iterable[LabeledSlideSampleBatch]:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self) -> Iterable[UnlabeledSlideSampleBatch]:
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
