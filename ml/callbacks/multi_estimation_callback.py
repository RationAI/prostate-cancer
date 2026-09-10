import tempfile
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
from sklearn.metrics import auc, roc_curve

from ml.typing import TilingSlideMetadata, UnlabeledTileSampleBatch


if TYPE_CHECKING:
    from ml.datamodule import TileDataModule


class MultiEstimationCallback(MultiloaderLifecycle):
    """Estimates hyper-parameters for several heuristic aggregators at once.

    Equivalent to running `EstimationCallback` once per aggregator, but the
    tile-level predict pass is shared: every aggregator's hyper-parameter
    grid is updated from the same batch, instead of re-running predict
    separately for each aggregator.

    For each aggregator, also scores every hyper-parameter configuration by
    AUC and logs all of them, plus the best-scoring configuration -- folding
    in what `postprocessing/eval_estimation.py` used to do as a separate
    offline step.
    """

    def __init__(
        self,
        aggregators: dict[str, dict[str, Any]],
    ) -> None:
        super().__init__()

        self.aggregator_cls: dict[str, type[Aggregator]] = {}
        self.static: dict[str, dict[str, int]] = {}
        self.param_names: dict[str, list[str]] = {}
        self.values_product: dict[str, list[tuple[int, ...]]] = {}

        for name, spec in aggregators.items():
            self.aggregator_cls[name] = get_class(spec["aggregator_cls_path"])
            self.static[name] = spec["static"]
            self.param_names[name] = list(spec["to_estimate"].keys())
            self.values_product[name] = list(product(*spec["to_estimate"].values()))

        self.tables: dict[str, dict[str, Any]] = {
            name: self._init_table(name) for name in aggregators
        }

    def _init_table(self, name: str) -> dict[str, Any]:
        table: dict[str, Any] = {"slide_name": [], "target": []}
        for values in self.values_product[name]:
            table[f"pred_{self._key(name, values)}"] = []
        return table

    def _key(self, name: str, values: tuple[int, ...]) -> str:
        return "_".join(
            f"{param_name}={value}"
            for param_name, value in zip(self.param_names[name], values, strict=True)
        )

    def on_predict_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        self.aggregators: dict[str, list[Aggregator]] = {}
        for name, values_product in self.values_product.items():
            self.aggregators[name] = [
                self.aggregator_cls[name](
                    **dict(zip(self.param_names[name], values, strict=True)),
                    **self.static[name],
                )
                for values in values_product
            ]

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

        for instances in self.aggregators.values():
            for aggregator in instances:
                aggregator.update(
                    preds=outputs,
                    targets=targets,
                    x=metadata["x"],
                    y=metadata["y"],
                )

    def on_predict_dataloader_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        slide_name = Path(self.slide["path"]).stem
        target = self.slide.get("carcinoma", None)

        for name, instances in self.aggregators.items():
            table = self.tables[name]
            table["slide_name"].append(slide_name)
            table["target"].append(target)

            for values, aggregator in zip(
                self.values_product[name], instances, strict=True
            ):
                pred, _ = aggregator.compute()
                table[f"pred_{self._key(name, values)}"].append(pred.item())

    def _compute_aucs(self, df: pd.DataFrame, name: str) -> dict[str, float]:
        # maps each configuration's sanitized key (mlflow metric names disallow "=") to its AUC
        aucs: dict[str, float] = {}
        for values in self.values_product[name]:
            column = f"pred_{self._key(name, values)}"
            fpr, tpr, _ = roc_curve(df["target"], df[column])
            aucs[self._key(name, values).replace("=", "_")] = float(auc(fpr, tpr))
        return aucs

    def on_predict_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        super().on_predict_epoch_end(trainer, pl_module)

        with tempfile.TemporaryDirectory() as tmp_dir:
            for name, table in self.tables.items():
                df = pd.DataFrame(table)
                filepath = Path(tmp_dir) / f"aggregated_predictions_{name}.json"
                df.to_json(filepath, orient="split")
                mlflow.log_artifact(str(filepath), artifact_path="tables")

                aucs = self._compute_aucs(df, name)
                mlflow.log_metrics(
                    {f"{name}/auc_{key}": value for key, value in aucs.items()}
                )

                best_key = max(aucs, key=lambda k: aucs[k])
                mlflow.log_metrics({f"{name}/best_auc": aucs[best_key]})
                mlflow.log_params({f"{name}/best_configuration": best_key})
