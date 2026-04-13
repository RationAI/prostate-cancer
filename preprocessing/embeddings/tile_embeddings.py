import os
from typing import cast

import albumentations as A
import hydra
import mlflow
import pandas as pd
import torch
from huggingface_hub import login
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocessing.embeddings.encoders import FoundationModel
from preprocessing.embeddings.merge_embeddings import attach_embeddings
from prostate_cancer.datamodule.datasets.tile_dataset import (
    SlideTiles,
    UnlabeledTilesDataset,
)


def compute_embeddings_slide(
    slide_dataset: SlideTiles,
    batch_size: int,
    device: torch.device,
    tile_encoder: FoundationModel,
) -> torch.Tensor:
    slide_dataloader = DataLoader(
        slide_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    slide_embeddings = torch.zeros(
        (len(slide_dataset), tile_encoder.embed_dim),
        device=device,
        dtype=torch.float32,
    )
    for i, (x, _) in enumerate(slide_dataloader):
        x = x.to(device)
        embeddings = cast("torch.Tensor", tile_encoder(x))  # (batch_size, embed_dim)

        start = i * batch_size
        end = start + embeddings.size(0)
        slide_embeddings[start:end] = embeddings

    return slide_embeddings


def compute_embeddings_uri(
    uri: str, config: DictConfig, tile_encoder: FoundationModel, device: torch.device
) -> tuple[pd.DataFrame, pd.DataFrame]:

    tiles = pd.read_parquet(mlflow.artifacts.download_artifacts(uri) + "/tiles.parquet")
    slides = pd.read_parquet(
        mlflow.artifacts.download_artifacts(uri) + "/slides.parquet"
    )

    dataset = UnlabeledTilesDataset(
        uris=(uri,),
        thresholds=config.thresholds,
        transforms=A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]  # Both PGP and Wirchow2 use the same normalization. This is also a default for Albumentation.
        ),
    )

    col = f"{tile_encoder.name}_embedding"
    tiles[col] = None

    partition = uri.split("/")[-1]
    for slide_dataset in tqdm(
        dataset.generate_datasets(), desc=f"Slides partition: {partition}"
    ):
        slide_embeddings = compute_embeddings_slide(
            slide_dataset, config.batch_size, device, tile_encoder
        )
        tiles = attach_embeddings(
            slide_embeddings, tiles, slide_dataset.slide_metadata, col
        )

    return slides, tiles


def compute_embeddings(
    tile_encoder: FoundationModel, config: DictConfig, device: torch.device
) -> None:
    with torch.no_grad():
        if isinstance(config.uris, DictConfig):
            uris = list(config.uris.values())
        else:
            uris = config.uris

        for uri in uris:
            partition = uri.split("/")[-1]
            slides, tiles = compute_embeddings_uri(uri, config, tile_encoder, device)
            save_mlflow_dataset(
                slides, tiles, f"{partition}_with_{tile_encoder.name}_embeddings"
            )


@with_cli_args(["+preprocessing=tile_embeddings"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    login(token=os.environ["HF_TOKEN"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_encoder: FoundationModel = hydra.utils.instantiate(config.tile_encoder)
    tile_encoder = tile_encoder.to(device)
    compute_embeddings(tile_encoder, config, device)


if __name__ == "__main__":
    main()
