from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as pl
import mlflow
import pandas as pd
import torch
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle
from rationai.mlkit.metrics.aggregators import Aggregator
from torchmetrics import (
    AUROC,
    Accuracy,
    NegativePredictiveValue,
    Precision,
    Recall,
    Specificity,
)

from ml.typing import (
    LabeledTileSampleBatch,
    TilingSlideMetadata,
    UnlabeledTileSampleBatch,
)


if TYPE_CHECKING:
    from ml.datamodule import TileDataModule


class MultiAggregatorEvalCallback(MultiloaderLifecycle):
    """Aggregates TL predictions into SL predictions with the max, mean_pool_max and top_k aggregators.

    Used in the test or predict stage.
    """

    def __init__(
        self,
        max_aggregator: Aggregator,
        mean_pool_max_aggregator: Aggregator,
        top_k_aggregator: Aggregator,
        max_threshold: float,
        mean_pool_max_threshold: float,
        top_k_threshold: float,
    ) -> None:
        super().__init__()
        self.aggregators_original = {
            "max": max_aggregator,
            "mean_pool_max": mean_pool_max_aggregator,
            "top_k": top_k_aggregator,
        }
        self.thresholds = {
            "max": max_threshold,
            "mean_pool_max": mean_pool_max_threshold,
            "top_k": top_k_threshold,
        }

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str | None = None,
    ) -> None:
        self.tables: dict[str, dict[str, Any]] = {
            name: {
                "slide_name": [],
                "prediction": [],
                "prediction_binary": [],
                "target": [],
            }
            for name in self.aggregators_original
        }
        self.majority_table: dict[str, Any] = {
            "slide_name": [],
            "prediction": [],
            "target": [],
        }

    def _on_dataloader_start(
        self, mode: str, trainer: pl.Trainer, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")
        # aggregators cannot be reset, thus their original state is copied for each slide
        self.aggregators = {
            name: deepcopy(aggregator)
            for name, aggregator in self.aggregators_original.items()
        }
        datamodule = cast("TileDataModule", trainer.datamodule)
        self.slide = cast(
            "TilingSlideMetadata", getattr(datamodule, mode).slides[dataloader_idx]
        )

    def on_test_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        self._on_dataloader_start("test", trainer, dataloader_idx)

    def on_predict_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        self._on_dataloader_start("predict", trainer, dataloader_idx)

    def _on_batch_end(
        self,
        outputs: torch.Tensor,
        batch: UnlabeledTileSampleBatch | LabeledTileSampleBatch,
    ) -> None:
        if len(batch) == 3:
            # Test step
            _, _, metadata = batch
        else:
            # Predict step
            _, metadata = batch

        targets = torch.zeros_like(outputs)
        for aggregator in self.aggregators.values():
            aggregator.update(
                preds=outputs,
                targets=targets,
                x=metadata["x"],
                y=metadata["y"],
            )

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: LabeledTileSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end(outputs, batch)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: UnlabeledTileSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end(outputs, batch)

    def _on_dataloader_end(self) -> None:
        slide_name = Path(self.slide["path"]).stem
        target = self.slide.get("carcinoma", None)

        votes = []
        for name, aggregator in self.aggregators.items():
            pred, _ = aggregator.compute()
            pred_binary = bool(pred.item() >= self.thresholds[name])
            votes.append(pred_binary)

            table = self.tables[name]
            table["slide_name"].append(slide_name)
            table["prediction"].append(pred.item())
            table["prediction_binary"].append(pred_binary)
            table["target"].append(target)

        self.majority_table["slide_name"].append(slide_name)
        self.majority_table["prediction"].append(sum(votes) >= 2)
        self.majority_table["target"].append(target)

    def on_test_dataloader_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        self._on_dataloader_end()

    def on_predict_dataloader_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        self._on_dataloader_end()

    def _log_tables_and_metrics(self) -> None:
        metrics: dict[str, float] = {}

        for name, table in self.tables.items():
            df = pd.DataFrame(table)
            filename = f"sl_predictions_{name}.json"
            df.to_json(filename, orient="split")
            mlflow.log_artifact(filename, artifact_path="tables")
            metrics.update(
                self._compute_metrics(
                    name,
                    table["target"],
                    table["prediction_binary"],
                    table["prediction"],
                )
            )

        majority_df = pd.DataFrame(self.majority_table)
        majority_df.to_json("sl_predictions_majority_vote.json", orient="split")
        mlflow.log_artifact("sl_predictions_majority_vote.json", artifact_path="tables")
        metrics.update(
            self._compute_metrics(
                "majority_vote",
                self.majority_table["target"],
                self.majority_table["prediction"],
                score=None,
            )
        )

        mlflow.log_metrics(metrics)

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        super().on_test_epoch_end(trainer, pl_module)
        self._log_tables_and_metrics()

    def on_predict_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        super().on_predict_epoch_end(trainer, pl_module)
        self._log_tables_and_metrics()

    @staticmethod
    def _compute_metrics(
        name: str,
        target: list[Any],
        prediction_binary: list[bool],
        score: list[float] | None = None,
    ) -> dict[str, float]:
        target_t = torch.tensor(target, dtype=torch.long)
        pred_binary_t = torch.tensor(prediction_binary, dtype=torch.long)

        binary_metrics = {
            "accuracy": Accuracy("binary"),
            "precision": Precision("binary"),
            "recall": Recall("binary"),
            "specificity": Specificity("binary"),
            "negative_predictive_value": NegativePredictiveValue("binary"),
        }
        results = {
            f"sl/{name}/{metric_name}": metric(pred_binary_t, target_t).item()
            for metric_name, metric in binary_metrics.items()
        }

        if score is not None:
            score_t = torch.tensor(score, dtype=torch.float)
            results[f"sl/{name}/AUC"] = AUROC("binary")(score_t, target_t).item()

        return results
