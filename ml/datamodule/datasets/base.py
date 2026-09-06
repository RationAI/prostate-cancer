import random
from abc import ABC
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar, cast

from albumentations.core.composition import TransformType
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets import MetaTiledSlides
from torch.utils.data import Dataset

from ml.typing import (
    LabeledTileSample,
    TilingSlideMetadata,
    UnlabeledTileSample,
)


T_co = TypeVar("T_co", covariant=True)


def get_slide_name(slide_metadata: TilingSlideMetadata) -> str:
    return Path(slide_metadata["path"]).stem


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


class BaseTileDataset(MetaTiledSlides[T_co]):
    """This class abstracts the functionality shared across embedding and image datasets."""

    def __init__(
        self,
        uris: Iterable[str],  # MLFlow URI(s) of tiled dataset
        single_slide_ds_cls: type[
            BaseSingleSlideDataset
        ],  # dataset class for tiles of a single slide
        carcinoma_roi_t: float | None = None,  # only for labeled
        stratified_filter: bool | None = None,  # only for labeled
        train_pos_tissue_roi_t: float
        | None = None,  # epithelium based training in labeled mode,
        transforms: TransformType | None = None,
        num_slides: int | None = None,  # cap slide count for very large datasets
        slide_range: tuple[int | None, int | None]
        | None = None,  # (start, end) slide index range, end inclusive; None bounds mean "from first"/"to last". Should be set only if want to select specific range of slides
    ) -> None:
        self.labeled = carcinoma_roi_t is not None and stratified_filter is not None
        self.train_pos_tissue_roi_t = train_pos_tissue_roi_t
        self.stratified_filter = stratified_filter
        self.carcinoma_roi_t = carcinoma_roi_t
        if (carcinoma_roi_t is None) ^ (stratified_filter is None):
            raise ValueError(
                "Either both should be None -> unlabeled mode, or both set -> labeled mode"
            )

        self.transforms = transforms
        self.single_slide_ds_cls = single_slide_ds_cls

        self.num_slides = num_slides
        self.slide_range = slide_range
        if num_slides is not None and slide_range is not None:
            raise ValueError(
                "Cannot use both deterministsic and non-deterministic subsampling"
            )

        super().__init__(uris=uris)

    def _slide_carcinoma_map(self) -> dict[str, bool]:
        return dict(
            zip(
                self.slides["id"],
                self.slides["carcinoma"],
                strict=True,
            )
        )

    def filter_non_carcinoma(self, tiles: HFDataset) -> HFDataset:
        """Filter negative tiles from positive slides."""
        assert self.labeled, "Only allowed for labeled dataset"

        slide_carcinoma = self._slide_carcinoma_map()

        def keep_row(row: dict[str, Any]) -> bool:
            is_pos_slide = slide_carcinoma[row["slide_id"]]

            # negative tiles in positive slides are filtered
            if is_pos_slide and not row["carcinoma"]:
                return False

            # breast training specific filter:
            # filter edge tiles which may contain wrongly detected epithelium
            return (
                self.train_pos_tissue_roi_t is None
                or (not is_pos_slide)
                or row["tissue_roi_percentage"] >= self.train_pos_tissue_roi_t
            )

        return tiles.filter(keep_row)

    def _subset_slides(
        self, slides: HFDataset, tiles: HFDataset, deterministic: bool
    ) -> tuple[HFDataset, HFDataset]:
        """Restricts slides/tiles to a uniform random or determinisitc sample of `self.num_slides`."""
        if deterministic:
            assert self.slide_range is not None
            start, end = self.slide_range
            start_idx = 0 if start is None else start
            stop_idx = len(slides) if end is None else end + 1  # end is inclusive

            if start_idx < 0 or stop_idx < 0 or stop_idx < start_idx:
                raise ValueError("Invalid bounds")

            selected_ids = set(slides["id"][start_idx:stop_idx])
        else:
            assert self.num_slides is not None
            selected_ids = set(random.sample(slides["id"], self.num_slides))

        slides = slides.filter(lambda row: row["id"] in selected_ids)
        tiles = tiles.filter(lambda row: row["slide_id"] in selected_ids)

        slides = slides.flatten_indices()
        tiles = tiles.flatten_indices()

        return slides, tiles

    def resample_slides(self) -> None:
        """Redraws a fresh random sample of `self.num_slides` slides.

        Rebuilds the underlying per-slide datasets in place (e.g. once per
        training epoch).
        """
        self.datasets = list(self.generate_datasets())
        self.cumulative_sizes = self.cumsum(self.datasets)

    def generate_datasets(self) -> Iterable[Dataset[T_co]]:
        # cache the full, unfiltered slides/tiles once so that repeated
        # (re)sampling always draws from the complete pool, not a previous subset
        if not hasattr(self, "_all_slides"):
            self._all_slides = self.slides
            self._all_tiles = self.tiles

        slides, tiles = self._all_slides, self._all_tiles

        if self.slide_range is not None:
            slides, tiles = self._subset_slides(slides, tiles, True)

        if self.num_slides is not None:
            slides, tiles = self._subset_slides(slides, tiles, False)

        self.slides = slides

        if self.labeled:
            # negative slides are never carcinoma, regardless of tile-level overlap
            # (e.g. epithelium tiles in negative slides are not carcinoma).
            # positive slides decide per-tile via carcinoma annotation (if present)
            # or epithelium annotation (weak substitute), thresholded.
            slide_carcinoma = self._slide_carcinoma_map()

            def label_row(row: dict[str, Any]) -> dict[str, bool]:
                # if negative slide, all its tiles are negative
                if not slide_carcinoma[row["slide_id"]]:
                    return {"carcinoma": False}

                # if positive slide, get the overlap (either epithelium or carcinoma)
                roi_percentage = (
                    row["carcinoma_roi_percentage"]
                    if "carcinoma_roi_percentage" in row
                    else row["epithelium_roi_percentage"]
                )

                # and threshold it
                return {"carcinoma": roi_percentage > self.carcinoma_roi_t}

            tiles = tiles.map(label_row)

            if self.stratified_filter:
                tiles = self.filter_non_carcinoma(tiles)

        # after this, global tiles are enhanced with carcinoma, possibly
        # filtered (if labeled stratified case), and possibly subset to fewer
        # slides -- the tile index is rebuilt to match (once per call, e.g.
        # on init and on each resample_slides())
        self.tiles = tiles
        self._meta.tiles = tiles
        self._meta._slide_id_to_indices = self._meta._build_tile_index(tiles)

        return (
            cast(
                "Dataset[T_co]",
                self.single_slide_ds_cls(
                    slide,
                    tiles=self._meta.filter_tiles_by_slide(slide["id"]),
                    include_label=self.labeled,
                    **({"transforms": self.transforms} if self.transforms else {}),
                ),
            )
            for slide in self.slides
        )
