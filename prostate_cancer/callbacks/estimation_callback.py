from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as pl
import mlflow
import pandas as pd
import torch
from hydra.utils import get_class
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle
from rationai.mlkit.metrics.aggregators import Aggregator

from prostate_cancer.typing import TilingSlideMetadata, UnlabeledTileSampleBatch


if TYPE_CHECKING:
    from prostate_cancer.datamodule import TileDataModule


class EstimationCallback(MultiloaderLifecycle):
    def __init__(
        self,
        aggregator_cls_path: str,
        to_estimate: dict[str, list[int]],
        static: dict[str, int],
    ) -> None:
        super().__init__()
        self.aggregator_cls: type[Aggregator] = get_class(aggregator_cls_path)
        self.to_estimate = to_estimate
        self.static = static
        self.param_names = list(to_estimate.keys())
        self.values_product = list(product(*self.to_estimate.values()))

        self.table: dict[str, Any] = {
            "slide_name": [],
            "target": [],
        }

        for values in self.values_product:
            keys = [
                f"{self.param_names[i]}={values[i]}"
                for i in range(len(self.to_estimate))
            ]

            key_str = "_".join(keys)
            self.table[f"pred_{key_str}"] = []

    def on_predict_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        self.aggregators = []
        for values in self.values_product:
            estimate_instance = dict(zip(self.param_names, values, strict=True))
            self.aggregators.append(
                self.aggregator_cls(
                    **estimate_instance,
                    **self.static,
                )
            )

        datamodule = cast("TileDataModule", trainer.datamodule)
        self.slide = cast(
            "TilingSlideMetadata", datamodule.predict.slides[dataloader_idx]
        )

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: UnlabeledTileSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, metadata = batch

        targets = torch.zeros_like(outputs)
        for aggregator in self.aggregators:
            aggregator.update(
                preds=outputs,
                targets=targets,
                x=metadata["x"],
                y=metadata["y"],
            )

    def on_predict_dataloader_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        # Compute the aggregated results for each kernel
        self.table["slide_name"].append(Path(self.slide["path"]).stem)

        for values, aggregator in zip(
            self.values_product, self.aggregators, strict=True
        ):
            pred, _ = aggregator.compute()
            keys = [
                f"{self.param_names[i]}={values[i]}"
                for i in range(len(self.to_estimate))
            ]
            key_str = "_".join(keys)
            self.table[f"pred_{key_str}"].append(pred.item())

        self.table["target"].append(self.slide.get("carcinoma", None))

    def on_predict_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        df = pd.DataFrame(self.table)
        df.to_json("aggregated_predictions.json", orient="split")
        mlflow.log_artifact(
            "aggregated_predictions.json",
            artifact_path="tables",
        )
