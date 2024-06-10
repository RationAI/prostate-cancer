# Copyright (c) The RationAI team.

import logging
from pathlib import Path

import lightning
import mlflow
import pyvips

from prostate_cancer.trainer.callbacks.dataloader_agnostic import (
    DataloaderAgnosticCallback,
)


logger = logging.getLogger("callbacks/cropped_heatmaps")


class CroppedHeatmaps(DataloaderAgnosticCallback):
    """A callback that creates heatmaps that are cropped to a list of masks.

    The callback requires the heatmap_visualizer callback to be registered. If no masks
    are entered, heatmaps are not generated.

    Attributes:
        save_dir: directory to which the cropped heatmaps will be saved
        masks: list of lists of format [uri, threshold]
        pred_dir: directory in which uncropped prediction heatmaps are stored
    """

    filename: str | None = None

    def __init__(
        self, save_dir: str, masks: dict[str, float] | None, pred_dir: str
    ) -> None:
        super().__init__()
        self.save_dir = Path(save_dir)
        self.masks = masks
        self.pred_dir = Path(pred_dir)
        self.mask_present = False

        if self.masks is None:
            return

        for uri in self.masks:
            self.mask_present = True
            if not isinstance(uri, str):
                raise TypeError(f"MLFlow URI must be a string, got {type(uri)}")
            if not uri.startswith("mlflow-artifacts:/"):
                raise ValueError(
                    f"MLFlow URI must start with 'mlflow-artifacts:/', got {uri}"
                )

        self.paths: list[Path] = []
        for uri in self.masks:
            self.paths.append(
                Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
            )

    def on_test_dataloader_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        metadata: dict,
        dataloader_idx: int,
    ) -> None:
        logger.debug("Creating new Cropped heatmap visualizer.")
        self.filename = metadata["slide_name"]

    def on_test_dataloader_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        dataloader_idx: int,
    ) -> None:
        if not self.mask_present:
            return

        pred_im = pyvips.Image.new_from_file(f"{self.pred_dir}/{self.filename}.tiff")
        masked_im = pred_im

        for idx, uri in enumerate(self.masks.keys()):
            threshold = self.masks[uri]
            mask_fp = next(self.paths[idx].glob(f"{self.filename}.*"))
            mask_im = pyvips.Image.new_from_file(mask_fp)
            if not (
                mask_im.width == pred_im.width and mask_im.height == pred_im.height
            ):
                logger.info(
                    f"Resolution of mask {idx} is not equal to the prediction mask resolution, mask {idx} will be rescaled"
                )
                mask_im = mask_im.resize(
                    pred_im.width / mask_im.width,
                    vscale=pred_im.height / mask_im.height,
                    kernel="nearest",
                )

            masked_im = masked_im * ((mask_im > (threshold * 255)) / 255)

        masked_im = masked_im.cast("uchar")

        logger.debug("Saving cropped heatmap.")
        save_path = Path(f"{self.save_dir}/{self.filename}.tiff")

        self.save_dir.mkdir(parents=True, exist_ok=True)

        masked_im.tiffsave(
            save_path,
            bigtiff=True,
            compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
            tile=True,
            tile_width=256,
            tile_height=256,
            pyramid=True,
        )

        mlflow.log_artifact(local_path=save_path, artifact_path=str(self.save_dir))
        artifact_uri = mlflow.get_artifact_uri(str(save_path))
        logger.debug(f"cropped heatmap saved to: {artifact_uri}")
        stripped_uri = artifact_uri.removeprefix("mlflow-artifacts:/")
        logger.debug(f"saving cropped heatmap URI to the cache as {stripped_uri}")
