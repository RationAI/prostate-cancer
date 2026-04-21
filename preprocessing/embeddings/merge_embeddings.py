"""Script to convert old way of handling embeddings (separate embeddings file per slide) to the new one where they are included in parquet files."""

from pathlib import Path

import hydra
import mlflow
import pandas as pd
import torch
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def attach_embeddings(
    slide_embeddings: torch.Tensor,
    tiles: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    embeds = slide_embeddings.cpu().numpy()

    if len(tiles) != len(embeds):
        raise ValueError(f"Mismatch: {len(tiles)} tiles vs {len(embeds)} embeddings")

    tiles[column] = list(embeds)
    return tiles


def process_and_shard_tiles(
    slides: pd.DataFrame,
    tiles: pd.DataFrame,
    slides_per_file: int,
    output_dir: Path,
    virchow2_embeddings_dir: Path | None,
    pgp_embeddings_dir: Path | None,
) -> None:
    for shard_idx in range(0, len(slides), slides_per_file):
        slides_chunk = slides.iloc[shard_idx : shard_idx + slides_per_file]

        tiles_buffer = []
        for (
            _,
            slide,
        ) in slides_chunk.iterrows():  # need for-loop because embed files are per slide
            slide_name = Path(slide.path).stem
            mask = tiles["slide_id"] == slide.id
            tiles_chunk = tiles.loc[mask].copy()

            if virchow2_embeddings_dir is not None:
                tiles_chunk["virchow2_embedding"] = None
            if pgp_embeddings_dir is not None:
                tiles_chunk["pgp_embedding"] = None

            if virchow2_embeddings_dir is not None:
                embeds = torch.load(
                    (virchow2_embeddings_dir / slide_name).with_suffix(".pt"),
                    map_location="cpu",
                )
                tiles_chunk = attach_embeddings(
                    embeds, tiles_chunk, "virchow2_embedding"
                )
                del embeds

            if pgp_embeddings_dir is not None:
                embeds = torch.load(
                    (pgp_embeddings_dir / slide_name).with_suffix(".pt"),
                    map_location="cpu",
                )
                tiles_chunk = attach_embeddings(embeds, tiles_chunk, "pgp_embedding")
                del embeds

            tiles_buffer.append(tiles_chunk)

        shard = pd.concat(tiles_buffer, ignore_index=True)
        shard.to_parquet(
            str(output_dir / f"tiles_{shard_idx:05d}.parquet"), index=False
        )
        tiles_buffer.clear()

        print(f"Saved shard {shard_idx:05d} with {len(shard)} tiles")
        del tiles_chunk


@with_cli_args(["+preprocessing=merge_embeddings"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tiling_path = mlflow.artifacts.download_artifacts(config.data.tiles_uri_224)
    slides = pd.read_parquet(tiling_path + "/slides.parquet")
    tiles = pd.read_parquet(tiling_path + "/tiles.parquet")

    virchow2_embeds_dir = (
        Path(mlflow.artifacts.download_artifacts(config.data.virchow2_embeddings_uri))
        if config.data.virchow2_embeddings_uri is not None
        else None
    )
    pgp_embeds_dir = (
        Path(mlflow.artifacts.download_artifacts(config.data.pgp_embeddings_uri))
        if config.data.pgp_embeddings_uri is not None
        else None
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slides_path = output_dir / "slides.parquet"
    slides.to_parquet(slides_path, index=False)  # slides.parquet is not changed

    process_and_shard_tiles(
        slides,
        tiles,
        config.slides_per_file,
        output_dir,
        virchow2_embeds_dir,
        pgp_embeds_dir,
    )

    mlflow.log_artifacts(str(output_dir), config.data.data_name + "_sharded")


if __name__ == "__main__":
    main()
