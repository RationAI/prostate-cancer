from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

import pandas as pd
from rationai.mlkit.data.datasets import MetaTiledSlides


T = TypeVar("T", covariant=True)


def get_slide_name(slide: pd.Series) -> str:
    return Path(slide.path).stem


class FilterableDataset(MetaTiledSlides[T]):
    """This class abstracts the filtering which is needed for all the datasets."""

    def __init__(
        self,
        uris: Iterable[str],
        thresholds: dict[str, float],
        carcinoma_roi_t: float | None = None,  # only for labeled
        stratified_filter: bool | None = None,  # only for labeled
    ) -> None:
        self.labeled = carcinoma_roi_t is not None and stratified_filter is not None
        self.stratified_filter = stratified_filter
        self.carcinoma_roi_t = carcinoma_roi_t
        self.thresholds = thresholds
        super().__init__(uris=uris)

    def filter_tiles_by_thresholds(self, tiles: pd.DataFrame) -> pd.DataFrame:
        for percentage in [
            "tissue_roi_percentage",
            "exclude_percentage",
            "another_pathology_percentage",
            "residual_percentage",
            "blur_percentage",
            "folding_percentage",
        ]:
            if percentage in tiles.columns:
                t = percentage.replace("percentage", "t")
                assert t in self.thresholds, f"{t} for {percentage}"
                mask = (
                    tiles[percentage] > self.thresholds[t]
                    if "tissue" in percentage
                    else tiles[percentage] <= self.thresholds[t]
                )
                tiles = tiles[mask]

        return tiles

    def prepare_tiles(self, tiles: pd.DataFrame) -> pd.DataFrame:
        assert self.labeled, "Only allowed for labeled dataset"
        tiles = self.filter_tiles_by_thresholds(tiles)
        tiles["carcinoma"] = tiles["carcinoma_roi_percentage"] > self.carcinoma_roi_t
        if self.stratified_filter:
            tiles = self.filter_non_carcinoma(tiles)

        return tiles

    def filter_non_carcinoma(self, tiles: pd.DataFrame) -> pd.DataFrame:
        assert self.labeled, "Only allowed for labeled dataset"
        tiles_slide_cancer = (
            tiles["slide_id"]
            .map(dict(zip(self.slides["id"], self.slides["carcinoma"], strict=True)))
            .astype(int)
        )

        return tiles[~((tiles_slide_cancer == 1) & (tiles["carcinoma"] == 0))]
