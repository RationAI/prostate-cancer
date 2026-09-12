from collections.abc import Iterable
from typing import TypeVar

import numpy as np
import torch
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from datasets import Dataset as HFDataset
from numpy.typing import NDArray
from openslide import OpenSlideCache
from ratiopath.openslide import OpenSlide
from rationai.mlkit.data.datasets import OpenSlideTilesDataset

from ml.datamodule.datasets.base import (
    BaseSingleSlideDataset,
    BaseTileDataset,
)
from ml.typing import (
    LabeledTileSample,
    TileMetadata,
    TilingSlideMetadata,
    UnlabeledTileSample,
)


T_co = TypeVar("T_co", covariant=True)

# Bytes of decoded-tile cache shared by every OpenSlide handle opened in this
# process. Without an explicit shared cache, each handle gets its own
# private cache of a (libopenslide-)default size; keeping many handles open
# at once (see CachedOpenSlideTilesDataset) then multiplies that private
# cache by the number of open slides, which is what caused the OOM. A single
# shared, size-capped cache keeps total decode memory bounded regardless of
# how many slides are open.
_SHARED_CACHE_BYTES = 256 * 1024 * 1024
_shared_cache: OpenSlideCache | None = None


def _get_shared_cache() -> OpenSlideCache:
    global _shared_cache
    if _shared_cache is None:
        _shared_cache = OpenSlideCache(_SHARED_CACHE_BYTES)
    return _shared_cache


class CachedOpenSlideTilesDataset(OpenSlideTilesDataset):
    """OpenSlideTilesDataset that keeps one lazily-opened handle for its
    slide instead of reopening the file on every `__getitem__` call.

    Reopening re-parses the pyramid/directory structure each time, which
    dominates runtime when tiles are sampled at random. Each instance only
    ever addresses a single slide_path, so a single cached handle per
    instance is sufficient -- no cross-slide eviction policy is needed for
    the handle itself. The handle is opened lazily (on first access, inside
    a DataLoader worker) rather than in `__init__`, since eagerly opening it
    in the main process before forking workers would share one native
    handle across processes.

    All handles opened by a worker process share one size-capped decode
    cache (see `_get_shared_cache`) instead of each getting its own private,
    unbounded-in-aggregate cache.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._handle: OpenSlide | None = None

    def __getitem__(self, idx: int) -> NDArray[np.uint8]:
        tile = self.tiles[idx]
        level = self._get_from_tile(tile, self.level)
        extent_x = self._get_from_tile(tile, self.tile_extent_x)
        extent_y = self._get_from_tile(tile, self.tile_extent_y)

        if self._handle is None:
            self._handle = OpenSlide(self.slide_path)
            self._handle.set_cache(_get_shared_cache())

        return self._handle.read_tile(tile["x"], tile["y"], extent_x, extent_y, level)

    def __del__(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._handle.close()


class TilesDataset(BaseTileDataset[T_co]):
    def __init__(
        self,
        uris: Iterable[str],
        carcinoma_roi_t: float | None = None,
        stratified_filter: bool | None = None,
        train_pos_tissue_roi_t: float | None = None,
        transforms: TransformType | None = None,
        num_slides: int | None = None,
        slide_range: tuple[int | None, int | None] | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(
            uris=uris,
            single_slide_ds_cls=SlideTiles,
            carcinoma_roi_t=carcinoma_roi_t,
            train_pos_tissue_roi_t=train_pos_tissue_roi_t,
            stratified_filter=stratified_filter,
            transforms=transforms,
            num_slides=num_slides,
            slide_range=slide_range,
        )


class LabeledTilesDataset(TilesDataset[LabeledTileSample]): ...


class UnlabeledTilesDataset(TilesDataset[UnlabeledTileSample]): ...


class SlideTiles(BaseSingleSlideDataset):
    def __init__(
        self,
        slide_metadata: TilingSlideMetadata,
        tiles: HFDataset,
        include_label: bool,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__(
            slide_metadata=slide_metadata,
            tiles=tiles,
            include_label=include_label,
        )
        self.slide_tiles = CachedOpenSlideTilesDataset(
            slide_path=slide_metadata["path"],
            level=slide_metadata["level"],
            tile_extent_x=slide_metadata["tile_extent_x"],
            tile_extent_y=slide_metadata["tile_extent_y"],
            tiles=tiles,
        )
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> LabeledTileSample | UnlabeledTileSample:
        image = self.slide_tiles[idx]

        tile_row = self.slide_tiles.tiles[idx]

        metadata = TileMetadata(
            slide=self.slide_tiles.slide_path.stem,
            x=tile_row["x"],
            y=tile_row["y"],
        )

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        tensor_image = self.to_tensor(image=image)["image"]

        if self.include_label:
            label = torch.tensor([tile_row["carcinoma"]]).float()
            return tensor_image, label, metadata

        return tensor_image, metadata
