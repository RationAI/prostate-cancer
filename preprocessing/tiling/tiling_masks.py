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


class PANDAHybridMask(PyvipsMask[TileMetadata]):
    background = 0
    stroma_connective_non_epithelium = 1
    healthy_epithelium = 2
    gleason_3 = 3
    gleason_4 = 4
    gleason_5 = 5

    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> TileMetadata:
        data = asdict(tile_labels)

        # tissue is all but background
        tissue = 1 - class_overlaps.get(self.background, 0)
        data["tissue_percentage"] = tissue

        # carcinoma is any type of carcinoma (overlaps are disjoint)
        data["carcinoma_percentage"] = (
            class_overlaps.get(self.gleason_3, 0)
            + class_overlaps.get(self.gleason_4, 0)
            + class_overlaps.get(self.gleason_5, 0)
        )
