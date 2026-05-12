"""Script for creating binary tile masks showing which tiles survive filtering."""

from pathlib import Path

import hydra
import pandas as pd
import ray
import torch
from omegaconf import DictConfig
from rationai.masks import process_items
from rationai.masks.mask_builders import ScalarMaskBuilder
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from prostate_cancer.datamodule.datasets.tile_dataset import UnlabeledTilesDataset


@ray.remote
def process_slide(slide: tuple[pd.DataFrame, pd.Series], output_path: Path) -> None:
    tiles, metadata = slide
    filename = Path(metadata.path).with_suffix(".tiff").name

    builder = ScalarMaskBuilder(
        output_path,
        filename,
        metadata.extent_x,
        metadata.extent_y,
        metadata.mpp_x,
        metadata.mpp_y,
        metadata.tile_extent_x,
        metadata.stride_x,
    )

    data = torch.ones(len(tiles)) * 255
    xs = torch.tensor(tiles["x"].values)
    ys = torch.tensor(tiles["y"].values)
    builder.update(data, xs, ys)

    builder.save()
    # ---


@with_cli_args(["+preprocessing=tile_survival"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    ds = UnlabeledTilesDataset(uris=config.tile_uris, thresholds=config.data.thresholds)
    items: list[tuple[pd.DataFrame, pd.Series]] = [
        (slide_ds.slide_tiles.tiles, slide_ds.slide_metadata)  # type: ignore[attr-defined]
        for slide_ds in ds.datasets
    ]
    output_path = Path(config.output_path)

    process_items(
        items,
        process_slide,
        fn_kwargs={
            "output_path": output_path,
        },
        max_concurrent=config.max_concurrent,
    )
    logger.log_artifacts(
        local_dir=str(output_path), artifact_path="tile_survival_masks"
    )


if __name__ == "__main__":
    main()
