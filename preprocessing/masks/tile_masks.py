"""Script for creating outlines of the tiles and also masks based on tiling percentages."""

from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import pyvips
import ray
import torch
from omegaconf import DictConfig
from rationai.masks import process_items, tile_mask, write_big_tiff
from rationai.masks.mask_builders import ScalarMaskBuilder
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@ray.remote
def process_slide(
    slide: pd.Series,
    percentage_cols: list[str],
    output_path: Path,
    tiles: pd.DataFrame,
) -> None:
    slide_tiles = tiles[tiles["slide_id"] == slide.id]

    # --- Percentage masks
    for percentage_col in percentage_cols:
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

        xs = torch.tensor(slide_tiles["x"].values)
        ys = torch.tensor(slide_tiles["y"].values)
        data = torch.tensor(slide_tiles[percentage_col].values)
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


@with_cli_args(["+preprocessing=tile_masks"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    paths = [mlflow.artifacts.download_artifacts(uri) for uri in config.tile_uris]
    slides = pd.read_parquet([Path(path) / "slides.parquet" for path in paths])
    tiles = pd.read_parquet([Path(path) / "tiles.parquet" for path in paths])

    output_path = Path(config.output_path)

    # Create subdirs for each tile mask type
    for percentage_col in [
        *config.percentage_cols,
        "outlines",
    ]:
        (output_path / percentage_col).mkdir(parents=True, exist_ok=True)

    process_items(
        slides.itertuples(),
        process_slide,
        fn_kwargs={
            "percentage_cols": config.percentage_cols,
            "output_path": output_path,
            "tiles": tiles,
        },
        max_concurrent=config.max_concurrent,
    )

    logger.log_artifacts(local_dir=str(output_path), artifact_path="tile_masks")


if __name__ == "__main__":
    main()
