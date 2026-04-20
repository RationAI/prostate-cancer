from collections.abc import Iterable
from pathlib import Path
from typing import TypeAlias, TypeVar, cast

import mlflow
import mlflow.artifacts
import pandas as pd
import torch
from torch.utils.data import Dataset

from prostate_cancer.datamodule.datasets.base import (
    FilterableDataset,
    get_slide_name,
)
from prostate_cancer.typing import LabeledSample, Metadata, UnlabeledSample


SlideMetadata: TypeAlias = pd.Series


T = TypeVar("T", covariant=True)


class EmbeddingsDataset(FilterableDataset[T]):
    """This dataset class wraps labeled gigapath features for all the slides in given URIs."""

    def __init__(
        self,
        uris: Iterable[str],
        embeddings_uri: str,
        thresholds: dict[str, float],
        carcinoma_roi_t: float | None = None,
        stratified_filter: bool | None = None,
    ) -> None:
        self.embeddings_folder = Path(
            mlflow.artifacts.download_artifacts(embeddings_uri)
        )
        super().__init__(
            uris=uris,
            thresholds=thresholds,
            carcinoma_roi_t=carcinoma_roi_t,
            stratified_filter=stratified_filter,
        )

    def generate_datasets(self) -> Iterable[Dataset[T]]:
        new_tiles = []
        datasets = []

        if self.labeled:
            self.tiles["carcinoma"] = (
                self.tiles["carcinoma_roi_percentage"] > self.carcinoma_roi_t
            )

        for _, slide in self.slides.iterrows():
            tiles, embeddings = self._filter_tiles_embeddings_by_slide(slide)
            if len(tiles) == 0:  # If there are no tiles, skip the slide
                print(f"Slide {get_slide_name(slide)} has no tiles. Skipping slide.")
                continue

            new_tiles.append(tiles)
            datasets.append(
                cast(
                    "Dataset[T]",
                    _TileEmbeddingsSlide(
                        slide_metadata=slide,
                        embeddings=embeddings,
                        tiles=tiles,
                        include_label=self.labeled,
                    ),
                ),
            )

        self.tiles = pd.concat(new_tiles, ignore_index=True)
        return datasets

    def _filter_tiles_embeddings_by_slide(
        self, slide: SlideMetadata
    ) -> tuple[pd.DataFrame, torch.Tensor]:
        slide_tiles = (
            self.tiles[self.tiles["slide_id"] == slide["id"]]
            .copy()
            .reset_index(drop=True)
        )
        slide_embeddings = cast(
            "torch.Tensor",
            torch.load(
                (self.embeddings_folder / get_slide_name(slide)).with_suffix(".pt"),
                map_location="cpu",
            ),
        )

        assert len(slide_tiles) == len(slide_embeddings), (
            "Tile and Embedding counts do not match"
        )

        slide_tiles = (
            self.prepare_tiles(slide_tiles)
            if self.labeled
            else self.filter_tiles_by_thresholds(slide_tiles)
        )
        slide_embeddings = slide_embeddings[slide_tiles.index.tolist()]
        return slide_tiles, slide_embeddings


class LabeledEmbeddingsDataset(EmbeddingsDataset[LabeledSample]): ...


class UnlabeledEmbeddingsDataset(EmbeddingsDataset[UnlabeledSample]): ...


class _TileEmbeddingsSlide(Dataset[LabeledSample | UnlabeledSample]):
    """This dataset class provides gigapath features for given slide (and optionally includes label)."""

    def __init__(
        self,
        slide_metadata: pd.Series,
        embeddings: torch.Tensor,
        tiles: pd.DataFrame,
        include_label: bool,
    ) -> None:
        super().__init__()
        self.include_label = include_label
        self.slide_metadata = slide_metadata
        self.tiles = tiles
        self.embeddings = embeddings

    def __len__(self) -> int:
        assert len(self.tiles) == len(self.embeddings)
        return len(self.tiles)

    def __getitem__(self, idx: int) -> LabeledSample | UnlabeledSample:
        vector = self.embeddings[idx]
        tile = self.tiles.iloc[idx]
        metadata = Metadata(
            slide=get_slide_name(self.slide_metadata), x=tile["x"], y=tile["y"]
        )

        if not self.include_label:
            return vector, metadata

        label = torch.tensor([self.tiles.iloc[idx]["carcinoma"]]).float()
        return vector, label, metadata
