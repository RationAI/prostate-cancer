from pathlib import Path
from typing import TYPE_CHECKING, cast

import mlflow
import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from rationai.masks.mask_builders import ScalarMaskBuilder
from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger

from prostate_cancer.typing import MILModelOutput, UnlabeledSlideSampleBatch


if TYPE_CHECKING:
    from prostate_cancer.datamodule import SlideDataModule


def min_max_normalization(tensor: torch.Tensor) -> torch.Tensor:
    weights_max = tensor.max()
    weights_min = tensor.min()
    return (tensor - weights_min) / (weights_max - weights_min)


class MILPredictionCallback(Callback):
    def get_mask_builder(
        self, slide_name: str, trainer: Trainer, save_dir: str
    ) -> ScalarMaskBuilder:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        datamodule = cast("SlideDataModule", trainer.datamodule)
        slides = cast("pd.DataFrame", datamodule.predict.slides)
        slides["name"] = slides["path"].apply(lambda x: Path(x).stem)

        _slide = slides[slides["name"] == slide_name]
        assert len(_slide) == 1
        slide = _slide.iloc[0]

        kwargs = {
            "save_dir": Path(save_dir),
            "filename": Path(slide.path).stem,
            "extent_x": slide.extent_x,
            "extent_y": slide.extent_y,
            "mpp_x": slide.mpp_x,
            "mpp_y": slide.mpp_y,
            "extent_tile": slide.tile_extent_x,
            "stride": slide.stride_x,
        }

        return ScalarMaskBuilder(**kwargs)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: MILModelOutput,
        batch: UnlabeledSlideSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        assert isinstance(trainer.logger, MLFlowLogger)
        sl_preds, tl_preds, batch_mask, batch_attention = outputs
        _, metadata_batch = batch

        # Log SL predictions
        trainer.logger.log_table(
            {
                "slide": [metadata["slide_name"] for metadata in metadata_batch],
                "sl_prediction": sl_preds.tolist(),
            },
            artifact_file="tables/sl_predictions.json",
        )

        # Log TL predictions and Attention Map
        for metadata, tl_preds_slide, mask_slide, attention_slide in zip(
            metadata_batch, tl_preds, batch_mask, batch_attention, strict=True
        ):
            for mask_type, data in zip(
                ["heatmaps", "attention_rescaled"],
                [tl_preds_slide, attention_slide],
                strict=True,
            ):
                mask_builder = self.get_mask_builder(
                    metadata["slide_name"], trainer, mask_type
                )
                data = data[mask_slide.bool()]  # take only real tiles (not padding)
                mask_builder.update(
                    min_max_normalization(data).cpu(), metadata["xs"], metadata["ys"]
                )
                mlflow.log_artifact(
                    str(mask_builder.save()), artifact_path=str(mask_builder.save_dir)
                )
