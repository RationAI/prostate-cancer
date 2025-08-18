"""Script to generate tissue masks for whole slide images (WSIs)."""

from pathlib import Path
from typing import cast

import hydra
import pyvips
import ray
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks import slide_resolution, tissue_mask, write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ray._private.worker import RemoteFunction0


def process_slide(slide_path: Path, level: int, output_path: Path) -> None:
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level=level)

    slide = cast("pyvips.Image", pyvips.Image.new_from_file(slide_path, level=level))
    mask = tissue_mask(slide, mpp=mpp_x)
    mask_path = output_path / slide_path.with_suffix(".tiff").name

    write_big_tiff(mask, path=mask_path, mpp_x=mpp_x, mpp_y=mpp_y)


def make_remote_process_slide(
    level: int, output_path: Path
) -> RemoteFunction0[None, Path]:
    @ray.remote
    def remote_process_slide(slide_path: Path) -> None:
        try:
            process_slide(slide_path, level, output_path)
        except Exception as e:
            print(f"Error processing slide {slide_path}: {e}")

    return remote_process_slide


@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    assert logger is not None, "Need logger"
    logger = cast("MLFlowLogger", logger)

    level = config.tissue_masks.level
    output_path = Path(config.tissue_masks.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    remote_process_slide = make_remote_process_slide(level, output_path)

    slides_path = Path(config.data_path).rglob("*.mrxs")
    test_slides_path = Path(config.test_data_path).rglob("*.mrxs")

    process_items(
        slides_path,
        process_item=remote_process_slide,
        max_concurrent=config.tissue_masks.max_concurrent,
    )

    process_items(
        test_slides_path,
        process_item=remote_process_slide,
        max_concurrent=config.tissue_masks.max_concurrent,
    )

    logger.experiment.log_artifacts(
        run_id=logger.run_id, local_dir=str(output_path), artifact_path="tissue_masks"
    )


if __name__ == "__main__":
    main()
