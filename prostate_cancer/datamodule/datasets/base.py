from abc import ABC
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar, cast

import mlflow
from albumentations.core.composition import TransformType
from datasets import Dataset as HFDataset, concatenate_datasets
from torch.utils.data import Dataset, ConcatDataset

from prostate_cancer.typing import (
    LabeledTileSample,
    TilingSlideMetadata,
    UnlabeledTileSample,
)


T = TypeVar("T", covariant=True)


def get_slide_name(slide_metadata: TilingSlideMetadata) -> str:
    return Path(slide_metadata.get("path")).stem


def download_artifacts(tiling_uris: Iterable[str]) -> tuple[HFDataset, HFDataset]:
    slide_dsets = []
    tile_dsets = []

    for tiling_uri in tiling_uris:
        root = Path(mlflow.artifacts.download_artifacts(tiling_uri))

        flat_slides = root / "slides.parquet"
        flat_tiles = root / "tiles.parquet"

        if flat_slides.exists():
            slide_dsets.append(HFDataset.from_parquet(str(flat_slides)))

        if flat_tiles.exists():
            tile_dsets.append(HFDataset.from_parquet(str(flat_tiles)))

        slide_folder = root / "slides"
        tile_folder = root / "tiles"

        slide_files = list(slide_folder.glob("*.parquet")) if slide_folder.exists() else []
        tile_files = list(tile_folder.glob("*.parquet")) if tile_folder.exists() else []

        if slide_files:
            slide_dsets.append(HFDataset.from_parquet([str(p) for p in slide_files]))

        if tile_files:
            tile_dsets.append(HFDataset.from_parquet([str(p) for p in tile_files]))

    if not slide_dsets:
        raise ValueError("No slide parquet files found in MLflow artifacts")

    if not tile_dsets:
        raise ValueError("No tile parquet files found in MLflow artifacts")

    slides = concatenate_datasets(slide_dsets)
    tiles = concatenate_datasets(tile_dsets)

    return slides, tiles



class BaseSingleSlideDataset(Dataset[LabeledTileSample | UnlabeledTileSample], ABC):
    def __init__(
        self,
        slide_metadata: TilingSlideMetadata,
        tiles: HFDataset,
        include_label: bool,
    ) -> None:
        super().__init__()
        self.include_label = include_label
        self.slide_metadata = slide_metadata
        self.tiles = tiles
        if len(tiles) == 0:
            print(
                f"Warning: No tiles found for slide {get_slide_name(slide_metadata)}."
            )


class BaseTileDataset(ConcatDataset[T]):
    """This class abstracts the functionality shared across embedding and image datasets."""

    slides: HFDataset
    tiles: HFDataset

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

        self.slides, self.tiles = download_artifacts(uris)
        self.tiles_by_slide: dict[bytes, list[int]] = {}
        for i, sid in enumerate(self.tiles["slide_id"]):
            self.tiles_by_slide.setdefault(sid, []).append(i)

        super().__init__(self.generate_datasets())

    def filter_non_carcinoma(self, tiles: HFDataset) -> HFDataset:
        assert self.labeled, "Only allowed for labeled dataset"

        slide_carcinoma = dict(
            zip(
                self.slides["id"],
                self.slides["carcinoma"],
                strict=True,
            )
        )

        return tiles.filter(
            lambda row: (
                not (slide_carcinoma[row["slide_id"]] == 1 and row["carcinoma"] == 0)
            )
        )

    def filter_tiles_by_slide(self, id: bytes) -> HFDataset:
        tile_indices = self.tiles_by_slide[id]
        slide_tiles = self.tiles.select(tile_indices)
        return slide_tiles

    def generate_datasets(self) -> Iterable[Dataset[T]]:

        if self.labeled:
            self.tiles = self.tiles.map(
                lambda row: {
                    "carcinoma": (
                        row["carcinoma_roi_percentage"] > self.carcinoma_roi_t
                    )
                }
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
                    ),
                ),
            )
            for slide in self.slides
        )
