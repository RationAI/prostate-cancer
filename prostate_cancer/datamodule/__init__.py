# Copyright (c) The RationAI team.

import logging
import sys
from collections import defaultdict
from copy import copy

import lightning.pytorch
from humanize import naturalsize
from torch.utils.data import DataLoader

from prostate_cancer.datamodule.datasets.base_wsi import BaseDataset
from prostate_cancer.datamodule.datasources import BaseDataSource


log = logging.getLogger("datamodule")


class WSIDataModule(lightning.pytorch.LightningDataModule):
    """WSIDataModule.

    Attributes:
        datasets (dict[str, BaseDataset]): Dict of datasets, keys must be in ['train', 'valid', 'test', 'predict'].
        data_sources (dict[str, BaseDataSource]):
        dataloaders_kwargs (dict[str, dict]): See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
    """

    datasets: dict[str, BaseDataset]
    data_sources: dict[str, BaseDataSource]
    dataloaders_kwargs: dict[str, dict]

    def __init__(
        self,
        datasets: dict[str, BaseDataset],
        data_sources: dict[str, BaseDataSource],
        dataloaders_kwargs: dict[str, dict],
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.data_sources = data_sources
        self.dataloaders_kwargs = dataloaders_kwargs
        self._built = False
        self._val_dataloader = None
        self._val_dataloader_built = False

    def setup(self, stage: str | None = None) -> None:
        if self._built:
            return

        splits = self._split_data_sources()
        self._build_datasets(splits)
        self._built = True

    def _split_data_sources(self) -> dict[str, dict[str, BaseDataSource]]:
        """Splits data sources into train, valid, test, and predict data sources.

        The original data sources are deleted after splitting.
        """
        all_ds_splits = defaultdict(dict)

        while self.data_sources:
            ds_name, ds = self.data_sources.popitem()
            ds_splits = ds.split()

            for stage, ds_split in ds_splits.items():
                all_ds_splits[stage][ds_name] = ds_split

        del self.data_sources
        log.debug(f"Data sources split between {', '.join(all_ds_splits.keys())}.")

        return dict(all_ds_splits)

    def _build_datasets(
        self, data_sources: dict[str, dict[str, BaseDataSource]]
    ) -> None:
        for stage, dataset in self.datasets.items():
            splits = data_sources.get(stage)
            if splits is None:
                raise ValueError(f"{stage} split not found in data sources.")

            if len(splits) > 1:
                raise NotImplementedError(
                    f"Multiple data sources found for {stage} split."
                )

            ds_name, ds = splits.popitem()
            dataset.sampler.build_inner_structure(ds)
            log.debug(f"{stage} dataset built")

    def _build_val_dataloader(self):
        """We typically rebuild the dataloaders after each epoch (in trainer by `reload_dataloaders_every_n_epochs=1`), but we want the validation dataloader unchanged."""
        self.datasets["valid"].generate_samples()
        kwargs = self.dataloaders_kwargs.get("valid", {})
        self._val_dataloader = DataLoader(self.datasets["valid"], **kwargs)

    def train_dataloader(self):
        self.datasets["train"].generate_samples()
        return DataLoader(
            self.datasets["train"], **self.dataloaders_kwargs.get("train", {})
        )

    def val_dataloader(self):
        if not self._val_dataloader_built:
            self._build_val_dataloader()
            self._val_dataloader_built = True
        return self._val_dataloader

    def predict_dataloader(self):
        self.datasets["predict"].generate_samples()
        return DataLoader(
            self.datasets["predict"], **self.dataloaders_kwargs.get("predict", {})
        )

    def test_dataloader(self):
        """On testing, we assume sequential sampling.

        One dataloader contains the tiles of one slide.
        We thus create a list of dataloaders, for the whole test set.
        todo: Current implementation requires large shm.
        """
        log.debug("Generating test DataLoaders")
        dataset = self.datasets["test"]
        dataloaders = []
        there_is_more = True
        while there_is_more:
            try:
                dataset.generate_samples()
                dataset_size = sys.getsizeof(dataset)
                dataset_size = naturalsize(dataset_size, binary=True)
                log.debug(f"Dataset samples generated ({dataset_size}).")
                dl = DataLoader(
                    copy(dataset), **self.dataloaders_kwargs.get("test", {})
                )
                dataloaders.append(dl)
                log.debug(
                    f"Dataloader copied (sampler.active.node={dataset.sampler.active_node})."
                )
            except StopIteration:
                there_is_more = False
        log.debug("Test Generators prepared")
        return dataloaders
