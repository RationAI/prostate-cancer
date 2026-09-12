from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from datasets import Dataset as HFDataset
from numpy.typing import NDArray
from openslide import OpenSlideCache
from rationai.mlkit.data.datasets import OpenSlideTilesDataset
from ratiopath.openslide import OpenSlide

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

# Bytes of decoded-tile pixel cache shared by every OpenSlide handle opened
# in this process (bounds decode memory; does NOT bound the per-handle
# directory/tile-offset metadata libopenslide keeps for as long as a handle
# stays open -- that is bounded separately by _MAX_OPEN_HANDLES below).
_SHARED_CACHE_BYTES = 256 * 1024 * 1024
_shared_cache: OpenSlideCache | None = None

# Max number of slide handles kept open per worker process at once. Keeping
# a handle open avoids reopening (and re-parsing the pyramid/directory
# structure of) the file on every tile read, but each open handle also
# holds real resident memory for that structure regardless of the decode
# cache size above -- for large pathology slides this was observed to be
# roughly ~10MB/handle. With num_slides=400 and no cap, up to num_workers x
# 400 handles could be open simultaneously, which is what pushed a ~20GiB
# run past 50GiB. This LRU cap bounds that to num_workers x _MAX_OPEN_HANDLES
# regardless of num_slides; tune it against your job's memory budget.
_MAX_OPEN_HANDLES = 32
_open_handles: OrderedDict[Path, OpenSlide] = OrderedDict()


def _get_shared_cache() -> OpenSlideCache:
    global _shared_cache
    if _shared_cache is None:
        _shared_cache = OpenSlideCache(_SHARED_CACHE_BYTES)
    return _shared_cache


def _get_handle(slide_path: Path) -> OpenSlide:
    handle = _open_handles.get(slide_path)
    if handle is not None:
        _open_handles.move_to_end(slide_path)
        return handle

    handle = OpenSlide(slide_path)
    handle.set_cache(_get_shared_cache())
    _open_handles[slide_path] = handle

    if len(_open_handles) > _MAX_OPEN_HANDLES:
        _, evicted = _open_handles.popitem(last=False)
        evicted.close()

    return handle


class CachedOpenSlideTilesDataset(OpenSlideTilesDataset):
    """Reuses a handle from a per-worker-process, size-capped LRU of open
    OpenSlide handles instead of reopening the slide file on every
    `__getitem__` call.

    Reopening re-parses the pyramid/directory structure each time, which
    dominates runtime when tiles are sampled at random across many slides.
    The cache is scoped to the worker process (module-level, populated
    lazily on first access inside a DataLoader worker -- never opened in the
    main process, since that would share one native handle across forked
    processes) and capped at `_MAX_OPEN_HANDLES` so memory stays bounded
    regardless of `num_slides`.
    """

    def __getitem__(self, idx: int) -> NDArray[np.uint8]:
        tile = self.tiles[idx]
        level = self._get_from_tile(tile, self.level)
        extent_x = self._get_from_tile(tile, self.tile_extent_x)
        extent_y = self._get_from_tile(tile, self.tile_extent_y)

        slide = _get_handle(self.slide_path)
        return slide.read_tile(tile["x"], tile["y"], extent_x, extent_y, level)


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
