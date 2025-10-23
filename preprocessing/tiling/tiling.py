"""Script for creating slides and tiles datasets for prostate cancer binary prediction."""

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import mlflow
import pandas as pd
import ray
from lightning.pytorch.loggers import Logger
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.tiling import tiling
from rationai.tiling.modules.tile_sources import OpenSlideTileSource
from rationai.tiling.modules.tile_sources.openslide_tile_source import OpenSlideMetadata
from rationai.tiling.typing import SlideMetadata, TiledSlideMetadata, TileMetadata
from rationai.tiling.writers import save_mlflow_dataset
from ray._private.worker import RemoteFunction0

from preprocessing.tiling.tiling_masks import (
    AnotherPathologyMask,
    BlurMask,
    CarcinomaMask,
    ExcludeMask,
    FoldingMask,
    ResidualMask,
    TissueMask,
)


# enhance slide metadata with cancer label
@dataclass
class CarcinomaOpenSlideMetadata(OpenSlideMetadata):
    carcinoma: bool | None


def process_slide(
    slide_path: Path,
    source: OpenSlideTileSource,
    blur_mask: BlurMask,
    folding_mask: FoldingMask,
    residual_mask: ResidualMask,
    tissue_mask: TissueMask,
    carcinoma_mask: CarcinomaMask,
    exclude_mask: ExcludeMask,
    another_path_mask: AnotherPathologyMask,
    tissue_masks_path: Path | None,
    qc_masks_path: Path | None,
    annotation_masks_path: Path | None,
    slide_label: bool | None,
) -> TiledSlideMetadata:
    slide_metadata, tiles = source(slide_path)

    tiff_slide_name = slide_path.with_suffix(".tiff").name

    if tissue_masks_path:
        tissue_mask_path = Path(tissue_masks_path) / tiff_slide_name
    else:
        tissue_mask_path = None

    if annotation_masks_path:
        carcinoma_mask_path = (
            Path(annotation_masks_path) / "carcinoma" / tiff_slide_name
        )
        exclude_mask_path = Path(annotation_masks_path) / "exclude" / tiff_slide_name
        another_path_mask_path = (
            Path(annotation_masks_path) / "another_pathology" / tiff_slide_name
        )
    else:
        carcinoma_mask_path = None
        exclude_mask_path = None
        another_path_mask_path = None

    if qc_masks_path:
        blur_mask_path = Path(qc_masks_path) / "blur_per_pixel" / tiff_slide_name
        folding_mask_path = Path(qc_masks_path) / "folding_per_pixel" / tiff_slide_name
        residual_mask_path = (
            Path(qc_masks_path) / "residual_per_pixel" / tiff_slide_name
        )
    else:
        blur_mask_path = None
        folding_mask_path = None
        residual_mask_path = None

    tiling_masks = [
        (tissue_mask, tissue_mask_path),
        (blur_mask, blur_mask_path),
        (folding_mask, folding_mask_path),
        (residual_mask, residual_mask_path),
        (exclude_mask, exclude_mask_path),
        (another_path_mask, another_path_mask_path),
        (carcinoma_mask, carcinoma_mask_path),
    ]

    for mask, mask_path in tiling_masks:
        if mask_path is not None and mask_path.exists():
            tiles = mask(mask_path, slide_metadata.extent, tiles)

    # if slide_label is None, this will result in undefined value in the carcinoma column
    slide_metadata = CarcinomaOpenSlideMetadata(
        **asdict(slide_metadata),
        carcinoma=slide_label,
    )

    return slide_metadata, tiles


def make_remote_process_slide(
    source: OpenSlideTileSource,
    blur_mask: BlurMask,
    folding_mask: FoldingMask,
    residual_mask: ResidualMask,
    tissue_mask: TissueMask,
    carcinoma_mask: CarcinomaMask,
    exclude_mask: ExcludeMask,
    another_path_mask: AnotherPathologyMask,
    tissue_masks_path: Path | None,
    qc_masks_path: Path | None,
    annotation_masks_path: Path | None,
    slide_labels: pd.DataFrame,
) -> RemoteFunction0[tuple[SlideMetadata, Iterable[TileMetadata]] | None, Path]:
    @ray.remote
    def remote_process_slide(
        slide_path: Path,
    ) -> tuple[SlideMetadata, Iterable[TileMetadata]] | None:
        try:
            label = (
                slide_labels.loc[str(slide_path), "carcinoma"]
                if str(slide_path) in slide_labels.index
                else None
            )
            return process_slide(
                slide_path,
                source,
                blur_mask,
                folding_mask,
                residual_mask,
                tissue_mask,
                carcinoma_mask,
                exclude_mask,
                another_path_mask,
                tissue_masks_path,
                qc_masks_path,
                annotation_masks_path,
                label,
            )
        except Exception as e:
            print(f"Error processing slide {slide_path}: {e}")
            return None

    return remote_process_slide


@hydra.main(
    config_path="../../configs", config_name="preprocessing_base", version_base=None
)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    tissue_masks_path = (
        None
        if config.tissue_masks_uri is None
        else Path(download_artifacts(config.tissue_masks_uri))
    )
    qc_masks_path = (
        None
        if config.qc_masks_uri is None
        else Path(download_artifacts(config.qc_masks_uri))
    )
    annotation_masks_path = (
        None
        if config.annotation_masks_uri is None
        else Path(download_artifacts(config.annotation_masks_uri))
    )

    # --- Source of raw tiles
    source = OpenSlideTileSource(
        mpp=config.mpp,
        tile_extent=config.tile_extent,
        stride=config.stride,
    )
    # ---

    # --- Mask instances

    # --- Half-extent (only ROI percentages are computed)
    carcinoma_mask = CarcinomaMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=config.tile_extent // 2,
        relative_roi_offset=0,
    )
    tissue_mask = TissueMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=config.tile_extent // 2,
        relative_roi_offset=0,
    )
    # ---

    # --- Full tile extent
    exclude_mask = ExcludeMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=config.tile_extent,
        relative_roi_offset=0,
    )
    another_path_mask = AnotherPathologyMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=config.tile_extent,
        relative_roi_offset=0,
    )
    blur_mask = BlurMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=config.tile_extent,
        relative_roi_offset=0,
    )
    folding_mask = FoldingMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=config.tile_extent,
        relative_roi_offset=0,
    )
    residual_mask = ResidualMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=config.tile_extent,
        relative_roi_offset=0,
    )
    # ---

    # ---

    slides_path = mlflow.artifacts.download_artifacts(config.slides_df_uri)
    slide_labels = pd.read_csv(slides_path).set_index("slide_path", drop=False)

    remote_process_slide = make_remote_process_slide(
        source,
        blur_mask,
        folding_mask,
        residual_mask,
        tissue_mask,
        carcinoma_mask,
        exclude_mask,
        another_path_mask,
        tissue_masks_path,
        qc_masks_path,
        annotation_masks_path,
        slide_labels,
    )

    slides = [Path(slide["slide_path"]) for _, slide in slide_labels.iterrows()]

    slides_df, tiles_df = tiling(slides=slides, handler=remote_process_slide)

    save_mlflow_dataset(
        slides=slides_df,
        tiles=tiles_df,
        dataset_name=config.data_name,
    )


if __name__ == "__main__":
    main()
