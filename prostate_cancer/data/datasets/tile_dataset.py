from collections.abc import Iterable
from typing import TypeVar, cast

import pandas as pd
import torch
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from rationai.mlkit.data.datasets import OpenSlideTilesDataset
from torch.utils.data import Dataset

from prostate_cancer.data.datasets.base import (
    FilterableDataset,
    get_slide_name,
)
from prostate_cancer.typing import LabeledSample, Metadata, UnlabeledSample


T = TypeVar("T", covariant=True)


class TilesDataset(FilterableDataset[T]):
    def __init__(
        self,
        uris: Iterable[str],
        thresholds: dict[str, float],
        carcinoma_roi_t: float | None = None,
        stratified_filter: bool | None = None,
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(
            uris=uris,
            thresholds=thresholds,
            carcinoma_roi_t=carcinoma_roi_t,
            stratified_filter=stratified_filter,
        )

    def generate_datasets(self) -> Iterable[Dataset[T]]:
        self.tiles = (
            self.prepare_tiles(self.tiles)
            if self.labeled
            else self.filter_tiles_by_thresholds(self.tiles)
        )
        return (
            cast(
                "Dataset[T]",
                SlideTiles(
                    slide,
                    tiles=self.filter_tiles_by_slide(slide["id"]),
                    include_label=self.labeled,
                    transforms=self.transforms,
                ),
            )
            for _, slide in self.slides.iterrows()
        )


class LabeledTilesDataset(TilesDataset[LabeledSample]): ...


class UnlabeledTilesDataset(TilesDataset[UnlabeledSample]): ...


class SlideTiles(Dataset[LabeledSample | UnlabeledSample]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        include_label: bool,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__()

        self.slide_tiles = OpenSlideTilesDataset(
            slide_path=slide_metadata.path,
            level=slide_metadata.level,
            tile_extent_x=slide_metadata.tile_extent_x,
            tile_extent_y=slide_metadata.tile_extent_y,
            tiles=tiles,
        )
        self.transforms = transforms
        self.include_label = include_label
        self.to_tensor = ToTensorV2()

        if len(tiles) == 0:
            print(
                f"Warning: No tiles found for slide {get_slide_name(slide_metadata.path)}."
            )

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> LabeledSample | UnlabeledSample:
        image = self.slide_tiles[idx]
        metadata = Metadata(
            slide=self.slide_tiles.slide_path.stem,
            x=self.slide_tiles.tiles.iloc[idx]["x"],
            y=self.slide_tiles.tiles.iloc[idx]["y"],
        )

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        tensor_image = self.to_tensor(image=image)["image"]

        if self.include_label:
            label = torch.tensor(
                [self.slide_tiles.tiles.iloc[idx]["carcinoma"]]
            ).float()
            return tensor_image, label, metadata

        return tensor_image, metadata
