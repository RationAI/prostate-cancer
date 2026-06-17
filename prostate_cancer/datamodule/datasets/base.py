from abc import ABC
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar, cast

import pandas as pd
from albumentations.core.composition import TransformType
from rationai.mlkit.data.datasets import MetaTiledSlides
from torch.utils.data import Dataset

from prostate_cancer.typing import LabeledTileSample, UnlabeledTileSample


T = TypeVar("T", covariant=True)


def get_slide_name(slide_metadata: pd.Series) -> str:
    return Path(slide_metadata.path).stem


class BaseSingleSlideDataset(Dataset[LabeledTileSample | UnlabeledTileSample], ABC):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        include_label: bool,
    ) -> None:
        super().__init__()
        assert "embedding" in tiles.column, (
            "Embeddings Dataset requires embedding column"
        )
        self.include_label = include_label
        self.slide_metadata = slide_metadata
        self.tiles = tiles
        if len(tiles) == 0:
            print(
                f"Warning: No tiles found for slide {get_slide_name(slide_metadata)}."
            )


class BaseTileDataset(MetaTiledSlides[T]):
    """This class abstracts the functionality shared across embedding and image datasets."""

    def __init__(
        self,
        uris: Iterable[str],
        single_slide_ds_cls: type[BaseSingleSlideDataset],
        carcinoma_roi_t: float | None = None,  # only for labeled
        stratified_filter: bool | None = None,  # only for labeled
        transforms: TransformType | None = None,
    ) -> None:
        self.labeled = carcinoma_roi_t is not None and stratified_filter is not None
        self.stratified_filter = stratified_filter
        self.carcinoma_roi_t = carcinoma_roi_t
        self.transforms = transforms
        self.single_slide_ds_cls = single_slide_ds_cls
        super().__init__(uris=uris)

    def filter_non_carcinoma(self, tiles: pd.DataFrame) -> pd.DataFrame:
        assert self.labeled, "Only allowed for labeled dataset"
        tiles_slide_cancer = (
            tiles["slide_id"]
            .map(dict(zip(self.slides["id"], self.slides["carcinoma"], strict=True)))
            .astype(int)
        )

        return tiles[~((tiles_slide_cancer == 1) & (tiles["carcinoma"] == 0))]

    def generate_datasets(self) -> Iterable[Dataset[T]]:

        if self.labeled:
            self.tiles["carcinoma"] = (
                self.tiles["carcinoma_roi_percentage"] > self.carcinoma_roi_t
            )
            if self.stratified_filter:
                self.tiles = self.filter_non_carcinoma(self.tiles)

        return (
            cast(
                "Dataset[T]",
                self.single_slide_ds_cls(
                    slide,
                    tiles=self.filter_tiles_by_slide(slide["id"]),
                    include_label=self.labeled,
                    **(
                        {"transforms": self.transforms}
                        if self.transforms is not None
                        else {}
                    ),  # avoid sending transforms arg to embeddings dataset
                ),
            )
            for _, slide in self.slides.iterrows()
        )
