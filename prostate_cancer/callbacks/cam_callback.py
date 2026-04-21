from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning
import lightning.pytorch as pl
import mlflow
import pandas as pd
import pyvips
import torch
from rationai.masks import write_big_tiff
from rationai.masks.mask_builders import TileMaskBuilder
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle
from torchvision.transforms import Resize

from prostate_cancer.cnn_model import CNNProstateModel
from prostate_cancer.typing import LabeledTileSampleBatch


if TYPE_CHECKING:
    from prostate_cancer.datamodule.data_module import TileDataModule
    from prostate_cancer.modeling.decode_head import BinaryClassifier


class CAMExplainer(MultiloaderLifecycle):
    def __init__(self, resize_shape: tuple[int, int]) -> None:
        super().__init__()
        self.resize = Resize(resize_shape)

    def on_test_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        if not isinstance(pl_module, CNNProstateModel):
            raise ValueError("Model must be a CNNProstateModel to generate CAMs.")

        self.model = pl_module
        self.decode_head = cast("BinaryClassifier", self.model.decode_head)
        self.linear_in_features = self.decode_head.proj.in_features

    def on_test_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        datamodule = cast("DataModule", trainer.datamodule)
        slide = cast("pd.Series", datamodule.test.slides.iloc[dataloader_idx])

        self.save_dir = "cam_explanations"
        self.mask_builder = TileMaskBuilder(
            save_dir=self.save_dir,
            filename=Path(slide.path).stem,
            extent_x=slide.extent_x,
            extent_y=slide.extent_y,
            mpp_x=slide.mpp_x,
            mpp_y=slide.mpp_y,
        )

    def on_test_dataloader_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        dataloader_idx: int,
    ) -> None:
        image_vips = pyvips.Image.new_from_array(self.mask_builder.image)
        count_vips = pyvips.Image.new_from_array(self.mask_builder.count)

        image_vips /= count_vips
        image_vips = 1.0 / (1.0 + (-image_vips).exp())  # sigmoid
        image_vips *= 255
        image_vips = image_vips.cast(pyvips.BandFormat.UCHAR)

        path = self.mask_builder.filename.with_suffix(".tiff")
        write_big_tiff(
            image_vips, path, self.mask_builder.mpp_x, self.mask_builder.mpp_y
        )

        mlflow.log_artifact(local_path=str(path), artifact_path=self.save_dir)
        path.unlink()

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: LabeledTileSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        inputs, _, metadata = batch
        data = self._explain(inputs)
        self.mask_builder.update(data=data, xs=metadata["x"], ys=metadata["y"])

    def _explain(self, inputs: torch.Tensor) -> torch.Tensor:
        feature_maps = self.model.backbone(inputs)  # (B, C, H, W)

        _, c, _, _ = feature_maps.shape
        linear_weights = self.decode_head.proj.weight.view(1, c, 1, 1)  # (1, C, 1, 1)
        cams = (feature_maps * linear_weights).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        return self.resize(cams.squeeze(1))  # (B, H_input, W_input)
