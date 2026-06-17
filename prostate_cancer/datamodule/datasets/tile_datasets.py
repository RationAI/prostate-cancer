from collections.abc import Iterable
from typing import TypeVar

import torch
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets import OpenSlideTilesDataset

from prostate_cancer.datamodule.datasets.base import (
    BaseSingleSlideDataset,
    BaseTileDataset,
)
from prostate_cancer.typing import LabeledTileSample, TileMetadata, UnlabeledTileSample, TilingSlideMetadata


T = TypeVar("T", covariant=True)


class TilesDataset(BaseTileDataset[T]):
    def __init__(
        self,
        uris: Iterable[str],
        carcinoma_roi_t: float | None = None,
        stratified_filter: bool | None = None,
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(
            uris=uris,
            single_slide_ds_cls=SlideTiles,
            carcinoma_roi_t=carcinoma_roi_t,
            stratified_filter=stratified_filter,
            transforms=transforms,
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
        self.slide_tiles = OpenSlideTilesDataset(
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
