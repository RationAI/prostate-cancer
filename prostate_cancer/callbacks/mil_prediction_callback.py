from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mlflow
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from lightning import Callback, LightningModule, Trainer
from rationai.masks.mask_builders import ScalarMaskBuilder
from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger

from prostate_cancer.typing import (
    LabeledBagOfTilesSampleBatch,
    MILModelOutput,
    UnlabeledBagOfTilesSampleBatch,
)


if TYPE_CHECKING:
    from prostate_cancer.datamodule import BagOfTilesDataModule


def min_max_normalization(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


class MILPredictionCallback(Callback):
    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: str | None = None
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        datamodule = cast("BagOfTilesDataModule", trainer.datamodule)
        dataset = datamodule.test if stage == "test" else datamodule.predict
        slides = cast("HFDataset", dataset.slides)

        self._slide_index = {
            Path(path).stem: i for i, path in enumerate(slides["path"])
        }

        self._slides = slides

        self.table: dict[str, Any] = {
            "slide": [],
            "sl_prediction": [],
        }

    def get_mask_builder(
        self,
        slide_name: str,
        trainer: Trainer,
        save_dir: str,
    ) -> ScalarMaskBuilder:

        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        slides = self._slides

        slide_idx = self._slide_index[slide_name]
        slide = slides[slide_idx]

        return ScalarMaskBuilder(
            save_dir=Path(save_dir),
            filename=Path(slide["path"]).stem,
            extent_x=slide["extent_x"],
            extent_y=slide["extent_y"],
            mpp_x=slide["mpp_x"],
            mpp_y=slide["mpp_y"],
            extent_tile=slide["tile_extent_x"],
            stride=slide["stride_x"],
        )

    def _on_batch_end(
        self,
        trainer: Trainer,
        outputs: MILModelOutput,
        batch: UnlabeledBagOfTilesSampleBatch | LabeledBagOfTilesSampleBatch,
    ) -> None:
        assert isinstance(trainer.logger, MLFlowLogger)

        sl_preds, tl_preds, batch_mask, batch_attention = outputs
        metadata_batch = batch[-1]

        self.table["slide"].extend([m["slide_name"] for m in metadata_batch])
        self.table["sl_prediction"].extend(sl_preds.tolist())

        for metadata, tl_preds_slide, mask_slide, attention_slide in zip(
            metadata_batch,
            tl_preds,
            batch_mask,
            batch_attention,
            strict=True,
        ):
            for mask_type, data in zip(
                ["heatmaps", "attention_rescaled"],
                [tl_preds_slide, attention_slide],
                strict=True,
            ):
                mask_builder = self.get_mask_builder(
                    metadata["slide_name"],
                    trainer,
                    mask_type,
                )

                # remove padded tiles
                data = data[mask_slide.bool()]

                mask_builder.update(
                    min_max_normalization(data).cpu(),
                    metadata["xs"],
                    metadata["ys"],
                )

                mlflow.log_artifact(
                    str(mask_builder.save()),
                    artifact_path=str(mask_builder.save_dir),
                )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: MILModelOutput,
        batch: LabeledBagOfTilesSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end(trainer, outputs, batch)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: MILModelOutput,
        batch: UnlabeledBagOfTilesSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end(trainer, outputs, batch)

    def _on_epoch_end(self) -> None:
        df = pd.DataFrame(self.table)
        df.to_json("sl_predictions.json", orient="split")
        mlflow.log_artifact(
            "sl_predictions.json",
            artifact_path="tables",
        )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._on_epoch_end()

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._on_epoch_end()
