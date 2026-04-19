"""Script to obtain binary masks from combined RGB PANDA mask."""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import pandas as pd
import pyvips
import ray
import tifffile
from mlflow.artifacts import download_artifacts
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.masks import write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide


@ray.remote(num_cpus=1, memory=(5 * 1024**3))
def process_slide(
    name: str, combined_masks_dir: Path, output_dir: Path, untied_masks: dict[str, list[int]]
) -> None:
    combined_mask_path = combined_masks_dir / f"{name}_mask.tiff"
    with OpenSlide(combined_mask_path) as slide:
        mpp_x, mpp_y = slide.slide_resolution(level=0)

    mask: NDArray[np.uint8] = tifffile.imread(combined_mask_path)
    mask = mask[..., 0]  # all information is stored in the first channel

    for binary_mask_name, cls_to_merge in untied_masks.items():
        nested_dir = output_dir / binary_mask_name
        nested_dir.mkdir(parents=True, exist_ok=True)

        binary_mask = np.isin(mask, cls_to_merge).astype(np.uint8) * 255

        output_path = nested_dir / f"{name}.tiff"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        write_big_tiff(
            image=pyvips.Image.new_from_array(binary_mask),
            path=output_path,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            tile_width=512,
            tile_height=512,
        )


@with_cli_args(["+preprocessing=untie_masks"])
@hydra.main(config_path="../../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = pd.read_csv(download_artifacts(config.data.metadata_table))

    with TemporaryDirectory() as output_dir:
        process_items(
            items=list(slides["slide_id"]),
            process_item=process_slide,
            fn_kwargs={
                "combined_masks_dir": Path(config.combined_masks_dir),
                "output_dir": Path(output_dir),
                "untied_masks": config.untied_masks,
            },
            max_concurrent=config.max_concurrent,
        )
        logger.log_artifacts(local_dir=output_dir)


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
