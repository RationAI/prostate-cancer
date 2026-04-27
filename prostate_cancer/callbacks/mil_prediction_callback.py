from pathlib import Path
from typing import TYPE_CHECKING, cast

import mlflow
import pandas as pd
from lightning import Callback, LightningModule, Trainer
from rationai.masks.mask_builders import ScalarMaskBuilder
from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger

from prostate_cancer.typing import MILModelOutput, UnlabeledSlideSampleBatch


if TYPE_CHECKING:
    from prostate_cancer.datamodule import SlideDataModule


class MILPredictionCallback(Callback):
    def get_mask_builder(self, slide_name: str, trainer: Trainer) -> ScalarMaskBuilder:
        datamodule = cast("SlideDataModule", trainer.datamodule)
        slides = cast("pd.DataFrame", datamodule.predict.slides)
        slides["name"] = slides["path"].apply(lambda x: Path(x).stem)

        _slide = slides[slides["name"] == slide_name]
        assert len(_slide) == 1
        slide = _slide.iloc[0]

        kwargs = {
            "save_dir": Path("heatmaps"),
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
        sl_preds, tl_preds = outputs

        # Log SL predictions
        trainer.logger.log_table(
            {
                "slide": [m["slide_name"] for m in batch[1]],
                "prediction": sl_preds.tolist(),
            },
            artifact_file="tables/sl_predictions.json",
        )

        # Log TL predictions
        slides_embeddings, metadata_batch = batch
        for slide_embeddings, xs, ys, slide_name, tl_preds_slide in zip(
            slides_embeddings,
            metadata_batch["xs"],
            metadata_batch["ys"],
            metadata_batch["slide_name"],
            tl_preds,
            strict=True,
        ):
            mask_builder = self.get_mask_builder(slide_name, trainer)
            slide_embeddings = slide_embeddings[
                : len(xs)
            ]  # take only real tiles (not padding)
            mask_builder.update(tl_preds_slide, xs, ys)

            mlflow.log_artifact(
                str(mask_builder.save()), artifact_path=str(mask_builder.save_dir)
            )
