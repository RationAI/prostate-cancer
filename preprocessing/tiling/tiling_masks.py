from dataclasses import asdict

from rationai.tiling.modules.masks import PyvipsMask
from rationai.tiling.typing import TileMetadata

from preprocessing.tiling.tiling_metadata import AugmentedTileMetadata


class TissueMask(PyvipsMask[TileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> TileMetadata | None:
        # drop empty tiles
        if class_overlaps.get(255, 0) > 0:
            data = asdict(tile_labels)
            data["tissue_roi_percentage"] = class_overlaps.get(255, 0)
            return AugmentedTileMetadata(**data)
        return None


class BlurMask(PyvipsMask[TileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> TileMetadata:
        data = asdict(tile_labels)
        data["blur_percentage"] = class_overlaps.get(255, 0)
        return AugmentedTileMetadata(**data)


class FoldingMask(PyvipsMask[TileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> TileMetadata:
        data = asdict(tile_labels)
        data["folding_percentage"] = class_overlaps.get(255, 0)
        return AugmentedTileMetadata(**data)


class ResidualMask(PyvipsMask[TileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> TileMetadata:
        data = asdict(tile_labels)
        data["residual_percentage"] = class_overlaps.get(255, 0)
        return AugmentedTileMetadata(**data)


class ExcludeMask(PyvipsMask[TileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> TileMetadata:
        data = asdict(tile_labels)
        data["exclude_percentage"] = class_overlaps.get(255, 0)
        return AugmentedTileMetadata(**data)


class AnotherPathologyMask(PyvipsMask[TileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> TileMetadata:
        data = asdict(tile_labels)
        data["another_pathology_percentage"] = class_overlaps.get(255, 0)
        return AugmentedTileMetadata(**data)


class CarcinomaMask(PyvipsMask[TileMetadata]):
    def forward_tile(
        self,
        tile_labels: TileMetadata,
        class_overlaps: dict[int, float],
    ) -> TileMetadata:
        data = asdict(tile_labels)
        data["carcinoma_roi_percentage"] = class_overlaps.get(255, 0)
        return AugmentedTileMetadata(**data)
