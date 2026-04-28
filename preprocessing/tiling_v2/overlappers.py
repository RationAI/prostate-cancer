from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from ratiopath.tiling.overlays import tile_overlay_overlap
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

    def _resolve_path(self, path: Path, fallback: str) -> str:
        target_path = self.mask_storage / f"{path.stem}.tiff"
        if target_path.exists():
            return str(target_path)

        return fallback  # not for all slides there are all masks, we return dummy black mask

    def add_mask_path_batch(self, batch: pd.DataFrame, fallback: str) -> pd.DataFrame:

        # assuming same name for the mask with .tiff suffix
        batch[f"{self.mask_name}_path"] = batch["path"].map(
            lambda x: self._resolve_path(Path(x), fallback)
        )
        return batch

    def add_mask_path(self, tiles: Dataset, fallback: str) -> Dataset:
        return tiles.map_batches(
            self.add_mask_path_batch,  # type: ignore[arg-type]
            fn_kwargs={"fallback": fallback},
            batch_format="pandas",
        )

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

    def filter(self, tiles: Dataset) -> Dataset:
        return tiles  # most of the overlappers do not filter


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
        tile[f"{self.mask_name}_percentage"] = (
            val if val is not None else 0.0
        )  # Can be None since dict is shared across slide
        return tile

    def add_percentages(self, tiles: Dataset) -> Dataset:
        tiles = self.add_overlaps(tiles)
        return tiles.map(self.extract_foreground_percentage)


# Is special because its the only one based on which we filter the tiles (and is always present)
class TissueOverlapper(BinaryOverlapper):
    def __init__(
        self,
        roi_corners: tuple[int, int, int, int],
        masks_folder: Path | None = None,
        masks_uri: str | None = None,
    ) -> None:
        super().__init__(
            roi_corners=roi_corners,
            mask_name="tissue_roi",
            masks_folder=masks_folder,
            masks_uri=masks_uri,
        )
        self.columns_to_keep |= {f"{self.mask_name}_percentage"}

    def filter(self, tiles: Dataset) -> Dataset:
        return tiles.filter(lambda row: row["tissue_roi_percentage"] > 0)
