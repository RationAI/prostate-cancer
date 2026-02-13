from collections.abc import Iterable
from typing import TypeAlias, cast

from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from rationai.mlkit.data.datasets import MetaTiledSlides
from torch.utils.data import DataLoader

from prostate_cancer.typing import LabeledSample, LabeledSampleBatch, UnlabeledSample


PartialConf: TypeAlias = DictConfig


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        sampler: PartialConf | None = None,
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
                    "MetaTiledSlides[LabeledSample]",
                    instantiate(self.datasets["train"]),
                )
                self.val = cast(
                    "MetaTiledSlides[LabeledSample]", instantiate(self.datasets["val"])
                )
            case "val":
                self.val = cast(
                    "MetaTiledSlides[LabeledSample]", instantiate(self.datasets["val"])
                )
            case "test":
                self.test = cast(
                    "MetaTiledSlides[LabeledSample]", instantiate(self.datasets["test"])
                )
            case "predict":
                self.predict = cast(
                    "MetaTiledSlides[UnlabeledSample]",
                    instantiate(self.datasets["predict"]),
                )

    def _load_sampler(
        self, dataset: MetaTiledSlides[LabeledSample]
    ) -> Iterable[LabeledSampleBatch] | None:
        if self.sampler_partial is not None:
            return instantiate(self.sampler_partial)(
                dataset=dataset, target_col="carcinoma"
            )

        return None

    def train_dataloader(self) -> Iterable[LabeledSampleBatch]:
        sampler = self._load_sampler(self.train)
        shuffle = (
            True if sampler is None else None
        )  # Sampler and shuffle are mutually exclusive
        return DataLoader(
            self.train,
            sampler=sampler,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> Iterable[LabeledSampleBatch]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> list[Iterable[LabeledSampleBatch]]:
        return [
            DataLoader(
                dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )
            for dataset in self.test.datasets
        ]

    def predict_dataloader(self) -> list[Iterable[LabeledSampleBatch]]:
        return [
            DataLoader(
                dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )
            for dataset in self.predict.datasets
        ]
