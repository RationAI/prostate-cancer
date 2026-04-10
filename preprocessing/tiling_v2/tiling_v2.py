from functools import partial
from typing import TYPE_CHECKING, Any

import hydra
import pandas as pd
import ray
import mlflow
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from rationai.mlkit import with_cli_args
from rationai.mlkit.autolog import autolog
from rationai.tiling.writers import save_mlflow_dataset
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles
from ratiopath.tiling.utils import row_hash


if TYPE_CHECKING:
    from preprocessing.tiling_v2.overlapers import Overlaper


ray.init(runtime_env={"excludes": [".git", ".venv"]})


def tile(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory tile generator."""
    return [
        {
            "tile_x": x,
            "tile_y": y,
            "path": row["path"],
            "slide_id": row["id"],
            "level": row["level"],
            "tile_extent_x": row["tile_extent_x"],
            "tile_extent_y": row["tile_extent_y"],
            "mpp_x": row["mpp_x"],
            "mpp_y": row["mpp_y"],
        }
        for x, y in grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
            stride=(row["stride_x"], row["stride_y"]),
        )
    ]


def filter_tissue(row: dict[str, Any]) -> bool:
    """Filter tiles to contain only tiles with some tissue."""
    return row["tissue_roi_percentage"] > 0


def select(row: dict[str, Any], to_keep: set[str]) -> dict[str, Any]:
    """Filter tiles to contain only desired columns."""
    tile = {
        "slide_id": row["slide_id"],
        "x": row["tile_x"],
        "y": row["tile_y"],
    }

    for column in to_keep:
        tile[column] = row[column]

    return tile


def carcinoma(row: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Add slide level carcinoma label to slides table."""
    slide_path = row["path"]
    match = df.loc[df["slide_path"] == slide_path, "carcinoma"]

    if match.empty:
        raise ValueError("No matching carcinoma label")

    row["carcinoma"] = match.iloc[0]
    return row


def tiling(df: pd.DataFrame, config: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlapers: list[Overlaper] = list(
        hydra.utils.instantiate(config.overlapers).values()
    )

    for overlaper in overlapers:
        overlaper.setup_storage()

    slides = read_slides(
        list(df["slide_path"]),
        tile_extent=config.tile_extent,
        stride=config.stride,
        mpp=config.mpp,
    )
    slides = slides.map(row_hash, num_cpus=0.1, memory=128 * 1024**2)
    slides = slides.map(partial(carcinoma, df=df), num_cpus=0.1, memory=128 * 1024**2)

    tiles = slides.flat_map(tile, num_cpus=0.2, memory=128 * 1024**2)
    tiles = tiles.repartition(target_num_rows_per_block=config.batch_size)

    to_keep = set()
    for overlaper in overlapers:
        tiles = overlaper.add_mask_path(tiles)
        tiles = overlaper.add_percentages(tiles)
        to_keep |= overlaper.columns_to_keep

    tiles = tiles.filter(filter_tissue, num_cpus=0.1, memory=128 * 1024**2)
    tiles = tiles.map(
        partial(select, to_keep=to_keep), num_cpus=0.1, memory=128 * 1024**2
    )

    return slides.to_pandas(), tiles.to_pandas()


@with_cli_args(["+preprocessing=tiling_v2"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    df = pd.read_csv(mlflow.artifacts.download_artifacts(config.data.metadata_table))
    slides, tiles = tiling(df, config)
    save_mlflow_dataset(slides, tiles, config.data.data_name)


if __name__ == "__main__":
    main()
