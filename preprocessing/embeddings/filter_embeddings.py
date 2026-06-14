"""Script for embeddings filtering based on tiles filtering (migration script)."""

from pathlib import Path
from typing import cast

import hydra
import mlflow
import pandas as pd
import torch
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from preprocessing.tiling_v2.filter_tiles import filter_tiles


def filter_and_log(
    embeddings_uri: str, tiling_uri: str, thresholds: dict[str, int], output_name: str
) -> None:

    embeds_path = Path(mlflow.artifacts.download_artifacts(embeddings_uri))
    tiling_path = Path(mlflow.artifacts.download_artifacts(tiling_uri))
    slides = pd.read_parquet(tiling_path / "slides.parquet")
    tiles = pd.read_parquet(tiling_path / "tiles.parquet")
    output_dir = Path(output_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, slide in slides.iterrows():
        slide_tiles = (
            tiles[tiles["slide_id"] == slide["id"]].copy().reset_index(drop=True)
        )
        slide_name = Path(slide.path).stem
        slide_embeddings = cast(
            "torch.Tensor",
            torch.load(
                (embeds_path / slide_name).with_suffix(".pt"),
                map_location="cpu",
            ),
        )
        assert len(slide_tiles) == len(slide_embeddings), (
            "Tile and Embedding counts do not match"
        )

        slide_tiles = filter_tiles(slide_tiles, thresholds)
        slide_embeddings = slide_embeddings[slide_tiles.index.tolist()]
        torch.save(slide_embeddings, (output_dir / slide_name).with_suffix(".pt"))


@with_cli_args(["+preprocessing=filter_embeddings"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    if (
        hasattr(config.data, "virchow2_embeddings_uri")
        and config.data.virchow2_embeddings_uri is not None
    ):
        filter_and_log(
            config.data.virchow2_embeddings_uri,
            config.data.tiles_uri_224,
            config.data.thresholds,
            config.data.data_name + "virchow2 embeddings",
        )

    if (
        hasattr(config.data, "pgp_embeddings_uri")
        and config.data.pgp_embeddings_uri is not None
    ):
        filter_and_log(
            config.data.pgp_embeddings_uri,
            config.data.tiles_uri_224,
            config.data.thresholds,
            config.data.data_name + "pgp embeddings",
        )


if __name__ == "__main__":
    main()
