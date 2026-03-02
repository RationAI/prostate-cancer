"""Script to generate annotation masks from XML files for whole slide images (WSIs)."""

from collections.abc import Iterable
from pathlib import Path
from typing import cast
from xml.etree import ElementTree as ET

import hydra
import mlflow
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from openslide import OpenSlide
from PIL.ImageDraw import _Ink
from rationai.masks import slide_resolution, write_big_tiff
from rationai.masks.annotations import XMLPolygonMask
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


class AnnotationMask(XMLPolygonMask):
    def __init__(
        self,
        path: str | Path,
        mask_size: tuple[int, int],
        mask_mpp_x: float,
        mask_mpp_y: float,
        annotation_mpp: tuple[float, float],
        group_name: str | None = None,
        mode: str = "P",
    ):
        super().__init__(path, mask_size, mask_mpp_x, mask_mpp_y, mode)
        self._annotation_mpp_x, self._annotation_mpp_y = annotation_mpp
        self.group_name = group_name

    @property
    def regions(self) -> Iterable[tuple[ET.Element, _Ink]]:
        for region in self.root.findall(".//Annotation"):
            if self.group_name is None or self.group_name == region.get("PartOfGroup"):
                yield region, 255

    def get_region_coordinates(
        self, region: ET.Element
    ) -> Iterable[tuple[float, float]]:
        for vertex in region.findall("Coordinates/Coordinate"):
            x_coord = vertex.get("X")
            y_coord = vertex.get("Y")
            if x_coord is None or y_coord is None:
                raise ValueError(
                    f"Invalid coordinate in XML: {vertex}. Expected 'X' and 'Y' attributes."
                )

            yield float(x_coord), float(y_coord)

    @property
    def annotation_mpp_x(self) -> float:
        return self._annotation_mpp_x

    @property
    def annotation_mpp_y(self) -> float:
        return self._annotation_mpp_y


@ray.remote
def process_slide(slide_path: Path, level: int, output_path: Path) -> None:
    ground_truth: str | int = slide_path.stem[-1]
    assert ground_truth in ("0", "1"), (
        f"Invalid slide name: {slide_path.stem}. Expected format: *-[0,1].mrxs"
    )
    ground_truth = int(ground_truth)

    # The annotation file is in the same directory as the slide
    annotation_file = slide_path.with_suffix(".xml")

    if not annotation_file.exists():
        return

    with OpenSlide(slide_path) as slide:
        tissue_mpp_x, tissue_mpp_y = slide_resolution(slide, level=level)
        annotation_mpp = slide_resolution(slide, level=0)
        mask_size = slide.level_dimensions[level]

    masks = [
        ("carcinoma", "Carcinoma"),
        ("exclude", "Exclude"),
        ("another_pathology", "Another pathology"),
    ]

    for mask_type, group_name in masks:
        annotator = AnnotationMask(
            path=annotation_file,
            mask_size=mask_size,
            mask_mpp_x=tissue_mpp_x,
            mask_mpp_y=tissue_mpp_y,
            annotation_mpp=annotation_mpp,
            group_name=group_name,
        )

        # Get the mask from the annotator
        mask = cast("pyvips.Image", pyvips.Image.new_from_array(annotator()))
        # If the mask is empty (all pixels are 0), skip saving
        if mask.max() == 0:
            continue

        # Save the mask to the destination directory
        mask_path = output_path / mask_type / slide_path.with_suffix(".tiff").name
        mask_path.parent.mkdir(exist_ok=True, parents=True)
        write_big_tiff(
            image=mask,
            path=mask_path,
            mpp_x=tissue_mpp_x,
            mpp_y=tissue_mpp_y,
        )


@with_cli_args(["+preprocessing=annot_masks"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    assert logger is not None, "Need logger"

    output_path = Path(config.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(mlflow.artifacts.download_artifacts(config.data.metadata_table))
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

    logger.log_artifacts(local_dir=str(output_path), artifact_path="annotation_masks")


if __name__ == "__main__":
    main()
