from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from ratiopath.tiling import tile_overlay_overlap
from ray.data import Dataset
from ray.data.expressions import col
from shapely.geometry import Polygon, box


class Overlapper(ABC):
    mask_storage: Path
    columns_to_keep: set[str]
    roi: Polygon

    def __init__(
        self,
        roi_corners: tuple[int, int, int, int],
        mask_name: str,
        masks_folder: Path | None = None,
        masks_uri: str | None = None,
    ) -> None:
        if (masks_folder is None and masks_uri is None) or (
            masks_folder is not None and masks_uri is not None
        ):
            raise ValueError("Provide exactly one source of masks")

        if masks_folder is not None:
            self.mask_storage = masks_folder
        else:
            self.mask_storage = Path(mlflow.artifacts.download_artifacts(masks_uri))

        self.roi = box(*roi_corners)
        self.mask_name = mask_name
        self.columns_to_keep = set()

    def add_mask_path_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        df = pd.DataFrame(batch)

        # assuming same name for the mask with .tiff suffix
        batch[f"{self.mask_name}_path"] = df["path"].map(
            lambda x: str(self.mask_storage / f"{Path(x).stem}.tiff")
        )
        return batch

    def add_mask_path(self, tiles: Dataset) -> Dataset:
        return tiles.map_batches(self.add_mask_path_batch)

    def add_overlaps(self, tiles: Dataset) -> Dataset:
        return tiles.with_column(
            f"{self.mask_name}_overlap",
            tile_overlay_overlap(
                roi=self.roi,
                overlay_path=col(f"{self.mask_name}_path"),
                tile_x=col("tile_x"),
                tile_y=col("tile_y"),
                mpp_x=col("mpp_x"),
                mpp_y=col("mpp_y"),
            ),
            num_cpus=1,
            memory=4 * 1024**3,
        )

    @abstractmethod
    def add_percentages(self, tiles: Dataset) -> Dataset: ...


class BinaryOverlapper(Overlapper):
    def __init__(
        self,
        roi_corners: tuple[int, int, int, int],
        mask_name: str,
        masks_folder: Path | None = None,
        masks_uri: str | None = None,
    ) -> None:
        super().__init__(
            roi_corners=roi_corners,
            mask_name=mask_name,
            masks_folder=masks_folder,
            masks_uri=masks_uri,
        )
        self.columns_to_keep |= {f"{self.mask_name}_percentage"}

    def extract_foreground_percentage(self, tile: dict[str, Any]) -> dict[str, Any]:
        val = tile[f"{self.mask_name}_overlap"].get("255", 0.0)
        tile[f"{self.mask_name}_percentage"] = val if val is not None else 0.0
        return tile

    def add_percentages(self, tiles: Dataset) -> Dataset:
        tiles = self.add_overlaps(tiles)
        return tiles.map(self.extract_foreground_percentage)
