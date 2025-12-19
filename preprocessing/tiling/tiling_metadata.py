from dataclasses import dataclass

from rationai.tiling.typing import TileMetadata


@dataclass
class AugmentedTileMetadata(TileMetadata):
    tissue_roi_percentage: float = 0.0
    blur_percentage: float = 0.0
    folding_percentage: float = 0.0
    residual_percentage: float = 0.0
    exclude_percentage: float = 0.0
    another_pathology_percentage: float = 0.0
    carcinoma_roi_percentage: float = 0.0
