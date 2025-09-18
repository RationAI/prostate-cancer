"""Script for creating slides and tiles datasets for prostate cancer binary prediction."""

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import pandas as pd
import ray
from lightning.pytorch.loggers import Logger
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.tiling import tiling
from rationai.tiling.modules.masks import PyvipsMask
from rationai.tiling.modules.tile_sources import OpenSlideTileSource
from rationai.tiling.modules.tile_sources.openslide_tile_source import OpenSlideMetadata
from rationai.tiling.typing import SlideMetadata, TiledSlideMetadata, TileMetadata
from rationai.tiling.writers import save_mlflow_dataset
from ray._private.worker import RemoteFunction0

from preprocessing.tiling.stratified_group_split import stratified_group_split


def carcinoma_bool(slide_path: Path | str) -> bool:
    """Get the slide cancer status from the slide name."""
    slide_path = Path(slide_path)

    if slide_path.stem[-1] not in ("0", "1"):
        raise ValueError(
            f"Invalid slide name: {slide_path.stem}. Expected format: *-[0,1].mrxs"
        )

    return slide_path.stem[-1] == "1"


# --- Stacking overlaps together (enforces ordering on mask application)
@dataclass
class TissueTileMetadata(TileMetadata):
    tissue_roi_percentage: float


@dataclass
class BlurTileMetadata(TissueTileMetadata):
    blur_percentage: float


@dataclass
class FoldingTileMetadata(BlurTileMetadata):
    folding_percentage: float


@dataclass
class ResidualTileMetadata(FoldingTileMetadata):
    residual_percentage: float


@dataclass
class ExcludeTileMetadata(ResidualTileMetadata):
    exclude_percentage: float


@dataclass
class AnotherPathologyTileMetadata(ExcludeTileMetadata):
    another_pathology_percentage: float


@dataclass
class CarcinomaTileMetadata(AnotherPathologyTileMetadata):
    carcinoma_roi_percentage: float


# ---


# enhance slide metadata with cancer label
@dataclass
class CarcinomaOpenSlideMetadata(OpenSlideMetadata):
    carcinoma: bool


class TissueMask(PyvipsMask[TissueTileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> TissueTileMetadata | None:
        # drop empty tiles
        if class_overlaps.get(255, 0) > 0:
            return TissueTileMetadata(
                **asdict(tile_labels), tissue_roi_percentage=class_overlaps.get(255, 0)
            )
        return None


class BlurMask(PyvipsMask[BlurTileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> BlurTileMetadata:
        return BlurTileMetadata(
            **asdict(tile_labels), blur_percentage=class_overlaps.get(255, 0)
        )


class FoldingMask(PyvipsMask[FoldingTileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> FoldingTileMetadata:
        return FoldingTileMetadata(
            **asdict(tile_labels), folding_percentage=class_overlaps.get(255, 0)
        )


class ResidualMask(PyvipsMask[ResidualTileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> ResidualTileMetadata:
        return ResidualTileMetadata(
            **asdict(tile_labels), residual_percentage=class_overlaps.get(255, 0)
        )


class ExcludeMask(PyvipsMask[ExcludeTileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> ExcludeTileMetadata:
        return ExcludeTileMetadata(
            **asdict(tile_labels), exclude_percentage=class_overlaps.get(255, 0)
        )


class AnotherPathologyMask(PyvipsMask[AnotherPathologyTileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> AnotherPathologyTileMetadata:
        return AnotherPathologyTileMetadata(
            **asdict(tile_labels),
            another_pathology_percentage=class_overlaps.get(255, 0),
        )


class CarcinomaMask(PyvipsMask[CarcinomaTileMetadata]):
    def forward_tile(
        self,
        tile_labels: TileMetadata,
        class_overlaps: dict[int, float],
    ) -> CarcinomaTileMetadata:
        return CarcinomaTileMetadata(
            **asdict(tile_labels), carcinoma_roi_percentage=class_overlaps.get(255, 0)
        )


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
    tissue_masks_path: Path,
    qc_masks_path: Path,
    annotation_masks_path: Path,
) -> TiledSlideMetadata | None:
    slide, tiles = source(slide_path)

    tiff_slide_name = slide_path.with_suffix(".tiff").name
    tissue_mask_path = Path(tissue_masks_path) / tiff_slide_name
    carcinoma_mask_path = Path(annotation_masks_path) / "carcinoma" / tiff_slide_name
    exclude_mask_path = Path(annotation_masks_path) / "exclude" / tiff_slide_name
    another_path_mask_path = (
        Path(annotation_masks_path) / "another_pathology" / tiff_slide_name
    )
    blur_mask_path = Path(qc_masks_path) / "blur_per_pixel" / tiff_slide_name
    folding_mask_path = Path(qc_masks_path) / "folding_per_pixel" / tiff_slide_name
    residual_mask_path = Path(qc_masks_path) / "residual_per_pixel" / tiff_slide_name

    # renaming necessary since types are different
    tissue_tiles = tissue_mask(tissue_mask_path, slide.extent, tiles)
    blur_tiles = blur_mask(blur_mask_path, slide.extent, tissue_tiles)
    folding_tiles = folding_mask(folding_mask_path, slide.extent, blur_tiles)
    residual_tiles = residual_mask(residual_mask_path, slide.extent, folding_tiles)

    if exclude_mask_path.exists():
        exclude_tiles = exclude_mask(exclude_mask_path, slide.extent, residual_tiles)
    else:
        exclude_tiles = [
            ExcludeTileMetadata(**asdict(tile), exclude_percentage=0.0)
            for tile in residual_tiles
        ]

    if another_path_mask_path.exists():
        another_path_tiles = another_path_mask(
            another_path_mask_path, slide.extent, exclude_tiles
        )
    else:
        another_path_tiles = [
            AnotherPathologyTileMetadata(
                **asdict(tile), another_pathology_percentage=0.0
            )
            for tile in exclude_tiles
        ]

    if carcinoma_mask_path.exists():
        carcinoma_tiles = carcinoma_mask(
            carcinoma_mask_path, slide.extent, another_path_tiles
        )
    # If it is positive slide and no carcinoma mask exists (there are no XML annotations) -> skip
    elif carcinoma_bool(slide_path):
        return None
    else:
        # Convert to CarcinomaTileMetadata with 0 cancer percentage if no carcinoma mask
        carcinoma_tiles = [
            CarcinomaTileMetadata(**asdict(tile), carcinoma_roi_percentage=0.0)
            for tile in another_path_tiles
        ]

    # --- postprocessing
    slide_carcinoma = carcinoma_bool(slide_path)
    slide = CarcinomaOpenSlideMetadata(**asdict(slide), carcinoma=slide_carcinoma)
    # ---

    return slide, carcinoma_tiles


def create_dataframe(slides: Iterable[Path]) -> pd.DataFrame:
    slides_df = pd.DataFrame(slides, columns=["path"])

    slides_df["carcinoma"] = slides_df["path"].apply(carcinoma_bool).astype(int)

    def get_case_id(slide_path: Path) -> str:
        # File names are expected to be in the format: "PREFIX-YEAR_CASEID-SLIDENUMBER-[0,1].mrxs"
        return slide_path.stem.split("_", 1)[1].split("-", 1)[0]

    slides_df["case_id"] = slides_df["path"].apply(get_case_id)

    return slides_df


def make_remote_process_slide(
    source: OpenSlideTileSource,
    blur_mask: BlurMask,
    folding_mask: FoldingMask,
    residual_mask: ResidualMask,
    tissue_mask: TissueMask,
    carcinoma_mask: CarcinomaMask,
    exclude_mask: ExcludeMask,
    another_path_mask: AnotherPathologyMask,
    tissue_masks_path: Path,
    qc_masks_path: Path,
    annotation_masks_path: Path,
) -> RemoteFunction0[tuple[SlideMetadata, Iterable[TileMetadata]] | None, Path]:
    @ray.remote
    def remote_process_slide(
        slide_path: Path,
    ) -> tuple[SlideMetadata, Iterable[TileMetadata]] | None:
        try:
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
            )
        except Exception as e:
            print(f"Error processing slide {slide_path}: {e}")
            return None

    return remote_process_slide


@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    # Download the tissue, annotation, and QC masks
    tissue_masks_path = Path(download_artifacts(config.tiling.tissue_masks_uri))
    annotation_masks_path = Path(download_artifacts(config.tiling.annotation_masks_uri))
    qc_masks_path = Path(download_artifacts(config.tiling.qc_masks_uri))

    # --- Source of raw tiles
    source = OpenSlideTileSource(
        mpp=config.mmci_mpp * (2**config.tiling.level),
        tile_extent=config.tiling.tile_extent,
        stride=config.tiling.stride,
    )
    # ---

    # --- Mask instances

    # --- Half-extent (only ROI percentages are computed)
    carcinoma_mask = CarcinomaMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=config.tiling.tile_extent // 2,
        relative_roi_offset=0,
    )
    tissue_mask = TissueMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=config.tiling.tile_extent // 2,
        relative_roi_offset=0,
    )
    # ---

    # --- Full tile extent
    exclude_mask = ExcludeMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=config.tiling.tile_extent,
        relative_roi_offset=0,
    )
    another_path_mask = AnotherPathologyMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=config.tiling.tile_extent,
        relative_roi_offset=0,
    )
    blur_mask = BlurMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=config.tiling.tile_extent,
        relative_roi_offset=0,
    )
    folding_mask = FoldingMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=config.tiling.tile_extent,
        relative_roi_offset=0,
    )
    residual_mask = ResidualMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=config.tiling.tile_extent,
        relative_roi_offset=0,
    )
    # ---

    # ---

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
    )

    # Load the slides
    slides_df = create_dataframe(Path(config.data_path).rglob("*.mrxs"))
    test_slides = create_dataframe(Path(config.test_data_path).rglob("*.mrxs"))

    # Split the dataset into train and validation sets
    train_slides, val_slides = stratified_group_split(
        data=slides_df,
        labels=slides_df["carcinoma"],
        groups=slides_df["case_id"],
        test_size=0.1,
        random_state=42,
    )

    # Log distribution of cancer slides
    print("Train slides:", train_slides["carcinoma"].value_counts())
    print("Validation slides:", val_slides["carcinoma"].value_counts())
    print("Test slides:", test_slides["carcinoma"].value_counts())

    # Tiling
    train_slides_df, train_tiles_df = tiling(
        slides=train_slides["path"], handler=remote_process_slide
    )
    val_slides_df, val_tiles_df = tiling(
        slides=val_slides["path"], handler=remote_process_slide
    )
    test_slides_df, test_tiles_df = tiling(
        slides=test_slides["path"], handler=remote_process_slide
    )

    save_mlflow_dataset(
        slides=train_slides_df,
        tiles=train_tiles_df,
        dataset_name="Prostate - train",
    )
    save_mlflow_dataset(
        slides=val_slides_df,
        tiles=val_tiles_df,
        dataset_name="Prostate - val",
    )
    save_mlflow_dataset(
        slides=test_slides_df,
        tiles=test_tiles_df,
        dataset_name="Prostate - test",
    )


if __name__ == "__main__":
    main()
