import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import albumentations as A
import hydra
import torch
from huggingface_hub import login
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from prostate_cancer.data.datasets import UnlabeledTilesDataset


if TYPE_CHECKING:
    from preprocessing.embeddings.encoders import FoundationModel


def save_embeddings(
    slide_embeddings: torch.Tensor,
    partition: str,
    slide_name: str,
    output_path: Path,
) -> None:
    folder = output_path / partition
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(slide_embeddings, (folder / slide_name).with_suffix(".pt"))


@hydra.main(
    config_path="../../configs",
    config_name="preprocessing/tile_embeddings",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    login(token=os.environ["HF_TOKEN"])
    dest = Path(config.output_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_encoder: FoundationModel = hydra.utils.instantiate(config.tile_encoder)
    tile_encoder = tile_encoder.to(device)

    with torch.no_grad():
        if isinstance(config.uris, DictConfig):
            uris = list(config.uris.values())
        else:
            uris = config.uris

        for uri in uris:
            dataset = UnlabeledTilesDataset(
                uris=(uri,),
                thresholds=config.thresholds,
                transforms=A.Compose(
                    [
                        A.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]  # Both PGP and Wirchow2 use the same normalization. This is also a default for Albumentation.
                ),
            )

            partition = uri.split("/")[-1]

            for slide_dataset in tqdm(
                dataset.generate_datasets(), desc=f"Slides partition: {partition}"
            ):
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

                    save_embeddings(slide_embeddings, partition, slide_name, dest)
                except Exception as e:
                    print(f"{e} occured during processing {slide_name}")

    logger.log_artifacts(local_dir=config.output_path)


if __name__ == "__main__":
    main()
