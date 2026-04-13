from pathlib import Path

import hydra
import mlflow
import pandas as pd
import torch
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset


def attach_embeddings(
    slide_embeddings: torch.Tensor,
    tiles: pd.DataFrame,
    slide: pd.Series,
    column: str,
) -> pd.DataFrame:
    embeds = slide_embeddings.cpu().numpy()
    mask = tiles["slide_id"] == slide.id

    if mask.sum() != len(embeds):
        raise ValueError(
            f"Mismatch for slide {slide.id}: {mask.sum()} tiles vs {len(embeds)} embeddings"
        )

    # to avoid pandas treating the arrays as multi-column scalars
    idx = tiles.index[mask]
    tiles.loc[mask, column] = pd.Series(list(embeds), index=idx)
    return tiles


def merge_embeddings(
    slides: pd.DataFrame, tiles: pd.DataFrame, embeddings_dir: Path, name: str
) -> pd.DataFrame:
    col = f"{name}_embedding"
    tiles[col] = None

    for _, slide in slides.iterrows():
        slide_name = Path(slide.path).stem
        embeds = torch.load(
            (embeddings_dir / slide_name).with_suffix(".pt"), map_location="cpu"
        )
        tiles = attach_embeddings(embeds, tiles, slide, col)

    return tiles


@with_cli_args(["+preprocessing=merge_embeddings"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tiling_path = mlflow.artifacts.download_artifacts(config.data.tiles_uri_224)

    virchow2_embeds_dir = (
        Path(mlflow.artifacts.download_artifacts(config.data.virchow2_embeddings_uri))
        if config.data.pgp_embeddings_uri is not None
        else None
    )
    pgp_embeds_dir = (
        Path(mlflow.artifacts.download_artifacts(config.data.pgp_embeddings_uri))
        if config.data.pgp_embeddings_uri is not None
        else None
    )
    slides = pd.read_parquet(tiling_path + "/slides.parquet")
    tiles = pd.read_parquet(tiling_path + "/tiles.parquet")

    if virchow2_embeds_dir is not None:
        tiles = merge_embeddings(slides, tiles, virchow2_embeds_dir, "virchow2")

    if pgp_embeds_dir is not None:
        tiles = merge_embeddings(slides, tiles, pgp_embeds_dir, "pgp")

    save_mlflow_dataset(slides, tiles, config.data.data_name + "_with_embeddings")


if __name__ == "__main__":
    main()
