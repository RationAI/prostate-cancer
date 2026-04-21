import os
from pathlib import Path
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
        embeddings = cast("torch.Tensor", tile_encoder(x))

        start = i * batch_size
        end = start + embeddings.size(0)
        slide_embeddings[start:end] = embeddings

    return slide_embeddings


def compute_embeddings_uri(
    uri: str,
    config: DictConfig,
    tile_encoder: FoundationModel,
    device: torch.device,
    output_dir: Path,
) -> Path:
    base_path = Path(mlflow.artifacts.download_artifacts(uri))

    tiles_all = pd.read_parquet(base_path / "tiles.parquet")
    slides = pd.read_parquet(base_path / "slides.parquet")

    dataset = UnlabeledTilesDataset(
        uris=(uri,),
        thresholds=config.thresholds,  # thresholds are set so that no tiles are filtered
        transforms=A.Compose(
            [
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        ),
    )

    partition = uri.split("/")[-1]
    partition_dir = output_dir / f"{partition}_with_{tile_encoder.name}_embeddings"
    partition_dir.mkdir(parents=True, exist_ok=True)
    slides_output = partition_dir / "slides"
    slides_output.mkdir(parents=True, exist_ok=True)
    tiles_output = partition_dir / "tiles"
    tiles_output.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    slides_in_shard = 0
    tiles_buffer = []

    for slide_dataset in tqdm(
        dataset.generate_datasets(),
        desc=f"Slides partition: {partition}",
    ):
        slide_embeddings = compute_embeddings_slide(
            slide_dataset, config.batch_size, device, tile_encoder
        )

        mask = tiles_all["slide_id"] == slide_dataset.slide_metadata.id
        tiles_chunk = tiles_all.loc[mask].copy()
        tiles_chunk = attach_embeddings(slide_embeddings, tiles_chunk)

        tiles_buffer.append(tiles_chunk)
        slides_in_shard += 1

        del slide_embeddings, tiles_chunk

        # flush shard
        if slides_in_shard >= config.slides_per_file:
            shard = pd.concat(tiles_buffer, ignore_index=True)

            shard.to_parquet(
                partition_dir / f"tiles_{shard_idx:05d}.parquet",
                index=False,
            )

            tiles_buffer.clear()
            slides_in_shard = 0
            shard_idx += 1

    # final flush
    if tiles_buffer:
        shard = pd.concat(tiles_buffer, ignore_index=True)
        shard.to_parquet(tiles_output / f"tiles_{shard_idx:05d}.parquet", index=False)
        tiles_buffer.clear()

    slides.to_parquet(slides_output / "slides.parquet", index=False)
    return partition_dir


def compute_embeddings(
    tile_encoder: FoundationModel,
    config: DictConfig,
    device: torch.device,
) -> None:
    with torch.no_grad():
        if isinstance(config.uris, DictConfig):
            uris = list(config.uris.values())
        else:
            uris = config.uris

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for uri in uris:
            out = compute_embeddings_uri(
                uri,
                config,
                tile_encoder,
                device,
                output_dir,
            )
            mlflow.log_artifacts(str(out), artifact_path=out.name)


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
