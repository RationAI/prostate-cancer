"""Script to rescale whole slide images (WSIs)."""

from math import isclose
from pathlib import Path
from typing import cast

import hydra
import mlflow
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks import slide_resolution, write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@ray.remote
def process_slide(slide_path: Path, output_path: Path, desired_mpp: float) -> None:
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level=0)

    assert isclose(mpp_x, mpp_y, rel_tol=0.1), f"{mpp_x} is not close to {mpp_y}"
    slide = cast("pyvips.Image", pyvips.Image.new_from_file(slide_path, level=0))
    scale_factor = mpp_x / desired_mpp
    print(f"Scale Factor for {slide_path.name}={scale_factor}")
    resized = slide.resize(scale_factor)
    resized_path = output_path / slide_path.with_suffix(".tiff").name

    write_big_tiff(resized, path=resized_path, mpp_x=desired_mpp, mpp_y=desired_mpp)
    print(f"Processed slide {slide_path.name}")


@with_cli_args(["+preprocessing=resize_wsi"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_path = Path(config.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(mlflow.artifacts.download_artifacts(config.slides_df_uri))
    slides_path = [Path(path) for path in df["slide_path"]]

    process_items(
        slides_path,
        fn_kwargs={"output_path": output_path, "desired_mpp": config.desired_mpp},
        process_item=process_slide,
        max_concurrent=config.max_concurrent,
    )

    logger.log_artifacts(
        config.output_path, artifact_path=f"resized_{config.desired_mpp}"
    )


if __name__ == "__main__":
    main()
