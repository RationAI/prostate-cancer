from abc import ABC
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar, cast

from albumentations.core.composition import TransformType
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets import MetaTiledSlides
from torch.utils.data import Dataset

from prostate_cancer.typing import LabeledTileSample, UnlabeledTileSample, TilingSlideMetadata


T = TypeVar("T", covariant=True)


def get_slide_name(slide_metadata: TilingSlideMetadata) -> str:
    return Path(slide_metadata.get("path")).stem


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


class BaseTileDataset(MetaTiledSlides[T]):
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
        super().__init__(uris=uris)

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
