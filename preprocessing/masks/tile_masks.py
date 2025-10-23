"""Script for creating outlines of the tiles and also masks based on tiling percentages."""

from pathlib import Path
from typing import Any, cast

import hydra
import mlflow
import numpy as np
import pandas as pd
import pyvips
import ray
import torch
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from rationai.masks import process_items, tile_mask, write_big_tiff
from rationai.masks.mask_builders import ScalarMaskBuilder
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ray._private.worker import RemoteFunction0


# This is used to create a filtered carcinoma mask for carcinoma threshold estimate by the pathologist
def filter_by_tissue(tiles: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return tiles[tiles["tissue_roi_percentage"] > threshold]


def process_slide(
    slide: Any,
    percentage_cols: list[str],
    output_path: Path,
    tissue_threshold: float,
    tiles_ref: Any,
) -> None:
    tiles = ray.get(tiles_ref)
    slide_tiles = tiles[tiles["slide_id"] == slide.id]

    # --- Percentage masks
    for percentage_col in [*percentage_cols, "carcinoma_filtered"]:
        should_filter = percentage_col == "carcinoma_filtered"
        filename = f"{Path(slide.path).stem}.tiff"
        save_dir = output_path / percentage_col

        builder = ScalarMaskBuilder(
            save_dir,
            filename,
            slide.extent_x,
            slide.extent_y,
            slide.mpp_x,
            slide.mpp_y,
            slide.tile_extent_x,
            slide.stride_x,
        )

        tiles_to_use = (
            filter_by_tissue(slide_tiles, tissue_threshold)
            if should_filter
            else slide_tiles
        )
        xs = torch.tensor(tiles_to_use["x"].values)
        ys = torch.tensor(tiles_to_use["y"].values)
        data = torch.tensor(
            tiles_to_use[
                percentage_col
                if percentage_col != "carcinoma_filtered"
                else "carcinoma_roi_percentage"
            ].values
        )
        builder.update(data, xs, ys)
        builder.save()
    # ---

    # --- Outlines
    mask = tile_mask(
        slide_tiles,
        tile_extent=(slide.tile_extent_x, slide.tile_extent_y),
        size=(slide.extent_x, slide.extent_y),
    )

    mask_path = output_path / "outlines" / f"{Path(slide.path).stem}.tiff"
    write_big_tiff(
        pyvips.Image.new_from_array(np.array(mask)),
        mask_path,
        mpp_x=slide.mpp_x,
        mpp_y=slide.mpp_y,
    )
    # ---


def make_remote_process_slide(
    percentage_cols: list[str],
    output_path: Path,
    tissue_threshold: float,
    tiles_ref: Any,
) -> RemoteFunction0[None, Path]:
    @ray.remote
    def remote_process_slide(slide_meta: Any) -> None:
        try:
            process_slide(
                slide_meta, percentage_cols, output_path, tissue_threshold, tiles_ref
            )
        except Exception as e:
            print(f"Error processing slide {slide_meta}: {e}")

    return remote_process_slide


@hydra.main(
    config_path="../../configs", config_name="preprocessing_base", version_base=None
)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    assert logger is not None, "Need logger"
    logger = cast("MLFlowLogger", logger)

    paths = [mlflow.artifacts.download_artifacts(uri) for uri in config.tile_uris]
    slides = pd.read_parquet([Path(path) / "slides.parquet" for path in paths])
    tiles = pd.read_parquet([Path(path) / "tiles.parquet" for path in paths])
    tiles_ref = ray.put(tiles)

    output_path = Path(config.output_path)

    # Create subdirs for each tile mask type
    for percentage_col in [
        *config.percentage_cols,
        "outlines",
        "carcinoma_filtered",
    ]:
        (output_path / percentage_col).mkdir(parents=True, exist_ok=True)

    remote_process_slide = make_remote_process_slide(
        config.percentage_cols,
        output_path,
        config.tissue_threshold,
        tiles_ref,
    )
    process_items(
        slides.itertuples(),
        remote_process_slide,
        max_concurrent=config.max_concurrent,
    )

    logger.log_artifacts(local_dir=str(output_path), artifact_path="tile_masks")


if __name__ == "__main__":
    main()
