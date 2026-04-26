"""These Datasets were taken from Adam Kukučka Ulcerative Colitis project and modified."""

from collections.abc import Iterable
from pathlib import Path
from typing import Generic, TypeVar

import mlflow
import mlflow.artifacts
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from prostate_cancer.datamodule.datasets.base import filter_tiles_by_thresholds
from prostate_cancer.typing import (
    LabeledSlideSample,
    SlideMetadata,
    UnlabeledSlideSample,
)


T = TypeVar("T", bound=LabeledSlideSample | UnlabeledSlideSample)


class SlideEmbeddingsDataset(Dataset[T], Generic[T]):
    def __init__(
        self,
        thresholds: dict[str, float],
        uris: Iterable[str],
        embeddings_uri: str,
        padding: bool = True,
        include_labels: bool = True,
    ) -> None:
        self.thresholds = thresholds
        self.include_labels = include_labels
        self.slides, self.tiles, self.embeddings_folder = self.download_artifacts(
            uris, embeddings_uri
        )
        self.padding = padding
        self.max_embeddings = self.tiles["slide_id"].value_counts().max()

    def download_artifacts(
        self, tiling_uris: Iterable[str], embeddings_uri: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
        slide_dfs = []
        tile_dfs = []

        for tiling_uri in tiling_uris:
            tiling_folder = Path(mlflow.artifacts.download_artifacts(tiling_uri))
            slide_dfs.append(pd.read_parquet(tiling_folder / "slides.parquet"))
            tile_dfs.append(pd.read_parquet(tiling_folder / "tiles.parquet"))

        embeddings_dir = Path(mlflow.artifacts.download_artifacts(embeddings_uri))

        return (
            pd.concat(slide_dfs, ignore_index=True),
            pd.concat(tile_dfs, ignore_index=True),
            embeddings_dir,
        )

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> T:
        slide_metadata = self.slides.iloc[idx]
        slide_name = Path(slide_metadata.path).stem
        slide_embeddings = torch.load(
            self.embeddings_folder / Path(slide_name).with_suffix(".pt"),
            map_location="cpu",
        )

        slide_tiles = self.tiles[self.tiles["slide_id"] == slide_metadata.id].reset_index(drop=True)
        assert len(slide_embeddings) == len(slide_tiles), "Size mismatch"
        filtered_tiles = filter_tiles_by_thresholds(slide_tiles, self.thresholds)
        slide_embeddings = slide_embeddings[filtered_tiles.index.tolist()]

        pad_amount = self.max_embeddings - slide_embeddings.shape[0]
        if self.padding:
            slide_embeddings = F.pad(slide_embeddings, (0, 0, 0, pad_amount), value=0.0)

        metadata = SlideMetadata(
            slide_id=slide_metadata["id"],
            slide_name=slide_name,
            slide_path=slide_metadata["path"],
        )

        if not self.include_labels:
            return slide_embeddings, metadata  # type: ignore[return-value]

        label = torch.tensor(slide_metadata.carcinoma).float()
        return slide_embeddings, label, metadata  # type: ignore[return-value]


class LabeledSlideEmbeddingsDataset(SlideEmbeddingsDataset[LabeledSlideSample]):
    def __init__(
        self,
        thresholds: dict[str, float],
        uris: Iterable[str],
        embeddings_uri: str,
        padding: bool = True,
    ) -> None:
        super().__init__(
            thresholds=thresholds,
            uris=uris,
            embeddings_uri=embeddings_uri,
            padding=padding,
            include_labels=True,
        )


class UnlabeledSlideEmbeddingsDataset(SlideEmbeddingsDataset[UnlabeledSlideSample]):
    def __init__(
        self,
        thresholds: dict[str, float],
        uris: Iterable[str],
        embeddings_uri: str,
        padding: bool = True,
    ) -> None:
        super().__init__(
            thresholds=thresholds,
            uris=uris,
            embeddings_uri=embeddings_uri,
            padding=padding,
            include_labels=False,
        )
