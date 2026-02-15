"""Script to generate tissue masks for whole slide images (WSIs)."""

from pathlib import Path
from typing import cast

import hydra
import mlflow
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks import slide_resolution, tissue_mask, write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@ray.remote
def process_slide(slide_path: Path, level: int, output_path: Path) -> None:
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level=level)

    slide = cast("pyvips.Image", pyvips.Image.new_from_file(slide_path, level=level))
    mask = tissue_mask(slide, mpp=mpp_x)
    mask_path = output_path / slide_path.with_suffix(".tiff").name

    write_big_tiff(mask, path=mask_path, mpp_x=mpp_x, mpp_y=mpp_y)


@with_cli_args(["+preprocessing=tissue_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_path = Path(config.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(mlflow.artifacts.download_artifacts(config.slides_df_uri))
    slides_path = [Path(path) for path in df["slide_path"]]

    process_items(
        slides_path,
        process_item=process_slide,
        fn_kwargs={
            "level": config.level,
            "output_path": output_path,
        },
        max_concurrent=config.max_concurrent,
    )

    logger.log_artifacts(local_dir=str(output_path), artifact_path="tissue_masks")


if __name__ == "__main__":
    main()
