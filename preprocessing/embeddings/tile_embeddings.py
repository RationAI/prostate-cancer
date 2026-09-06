import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import albumentations as A
import hydra
import torch
from huggingface_hub import login
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.datamodule.datasets import UnlabeledTilesDataset


if TYPE_CHECKING:
    from ml.modeling.backbone.foundation_base import FoundationModel


@with_cli_args(["+preprocessing=tile_embeddings"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    login(token=os.environ["HF_TOKEN"])
    dest = Path(config.output_path)
    dest.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_encoder: FoundationModel = hydra.utils.instantiate(config.tile_encoder)
    tile_encoder = tile_encoder.to(device)

    tiling_uri = config.data.tiles_filtered_uri_224
    slide_range = (
        None
        if config.start is None and config.end is None
        else (config.start, config.end)
    )

    with torch.no_grad():
        dataset = UnlabeledTilesDataset(
            uris=(tiling_uri,),
            transforms=A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]  # Both PGP and Wirchow2 use the same normalization. This is also a default for Albumentation.
            ),
            slide_range=slide_range,
        )

        for slide_dataset in tqdm(dataset.datasets):
            slide_name = Path(slide_dataset.slide_tiles.slide_path).stem
            try:
                slide_dataloader = DataLoader(
                    slide_dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                )
                slide_embeddings = torch.zeros(
                    (len(slide_dataset), tile_encoder.embed_dim),
                    device=device,
                    dtype=torch.float32,
                )
                for i, (x, _) in enumerate(slide_dataloader):
                    x = x.to(device)
                    embeddings = cast(
                        "torch.Tensor", tile_encoder(x)
                    )  # (batch_size, embed_dim)

                    start = i * config.batch_size
                    end = start + embeddings.size(0)
                    slide_embeddings[start:end] = embeddings

                torch.save(slide_embeddings, (dest / slide_name).with_suffix(".pt"))

            except Exception as e:  # noqa: BLE001
                print(f"{e} occured during processing {slide_name}")

    logger.log_artifacts(local_dir=config.output_path)


if __name__ == "__main__":
    main()
