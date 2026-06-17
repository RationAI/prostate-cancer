"""Script for tiles filtering based on estimated thresholds of overlaps with masks."""

from pathlib import Path

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset


def filter_tiles(tiles: pd.DataFrame, thresholds: dict[str, int]) -> pd.DataFrame:
    for col in tiles.columns:
        if col.endswith("percentage") and "carcinoma" not in col:
            t = col.replace("percentage", "t")
            assert t in thresholds, f"{t} for {col}"
            mask = (
                tiles[col] > thresholds[t]
                if "tissue" in col
                else tiles[col] <= thresholds[t]
            )
            tiles = tiles[mask]

    return tiles


def filter_and_log(
    tiling_uri: str, thresholds: dict[str, int], dataset_name: str
) -> None:
    tiling_path = Path(mlflow.artifacts.download_artifacts(tiling_uri))
    slides = pd.read_parquet(tiling_path / "slides.parquet")
    tiles = pd.read_parquet(tiling_path / "tiles.parquet")
    tiles = filter_tiles(tiles, thresholds)
    save_mlflow_dataset(slides, tiles, dataset_name)


@with_cli_args(["+preprocessing=filter_tiles"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    if hasattr(config.data, "tiles_uri_512") and config.data.tiles_uri_512 is not None:
        filter_and_log(
            config.data.tiles_uri_512,
            config.data.thresholds,
            config.data.data_name + "_512",
        )

    if hasattr(config.data, "tiles_uri_224") and config.data.tiles_uri_224 is not None:
        filter_and_log(
            config.data.tiles_uri_224,
            config.data.thresholds,
            config.data.data_name + "_224",
        )


if __name__ == "__main__":
    main()
