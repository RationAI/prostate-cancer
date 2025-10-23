from collections.abc import Iterable
from pathlib import Path
from typing import cast

import albumentations as A
import hydra
import timm
import torch
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from prostate_cancer.data.datasets import UnlabeledTilesDataset


def load_dataset(
    thresholds: dict[str, float], uris: Iterable[str]
) -> UnlabeledTilesDataset:
    transforms = A.Compose([A.Normalize()])
    return UnlabeledTilesDataset(
        uris=uris,
        thresholds=thresholds,
        transforms=transforms,
    )


PGP_EMBEDDING_DIM = 1536


def load_tile_encoder() -> torch.nn.Module:
    """Function to fetch the tile encoder model from HuggingsFace.

    Note:
        For this, you need to setup HF_TOKEN=<X> env.variable.
    """
    return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)


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
    config_path="../../configs", config_name="preprocessing_base", version_base=None
)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    assert logger is not None, "Need logger"
    logger = cast("MLFlowLogger", logger)
    dest = Path(config.output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_encoder = load_tile_encoder().to(device).eval()

    with torch.no_grad():
        for uri in config.uris:
            dataset = load_dataset(config.thresholds, (uri,))

            partition = uri.split(" - ")[-1]

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
                        (len(slide_dataset), PGP_EMBEDDING_DIM),
                        device=device,
                        dtype=torch.float32,
                    )
                    for i, (x, _) in enumerate(slide_dataloader):
                        x = x.to(device)
                        embeddings = cast(
                            "torch.Tensor", tile_encoder(x)
                        )  # (batch_size, PGP_EMBEDDING_DIM)

                        start = i * config.batch_size
                        end = start + embeddings.size(0)
                        slide_embeddings[start:end] = embeddings

                    save_embeddings(slide_embeddings, partition, slide_name, dest)
                except Exception as e:
                    print(f"{e} occured during processing {slide_name}")

    logger.experiment.log_param(logger.run_id, "model", "prov-gigapath")
    logger.experiment.log_param(logger.run_id, "save_destination", config.output_path)

    logger.log_artifacts(
        local_dir=config.output_path,
    )


if __name__ == "__main__":
    main()
