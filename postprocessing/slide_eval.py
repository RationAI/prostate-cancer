import json
from typing import cast

import hydra
import mlflow
import pandas as pd
import torch
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torchmetrics import (
    AUROC,
    Accuracy,
    NegativePredictiveValue,
    Precision,
    Recall,
    Specificity,
)


def evaluate(table: pd.DataFrame, t: float) -> dict[str, float]:
    metrics = {
        "AUC": AUROC("binary"),
        "accuracy": Accuracy("binary"),
        "precision": Precision("binary"),
        "recall": Recall("binary"),
        "specificity": Specificity("binary"),
        "negative_predictive_value": NegativePredictiveValue("binary"),
    }

    target = torch.tensor(table["target"].astype(int).values)
    pred_prob = torch.tensor(table["prediction"].astype(float).values)
    pred_label = (pred_prob >= t).int()

    results = {}
    for name, metric in metrics.items():
        if name == "AUC":
            value = metric(pred_prob, target)
        else:
            value = metric(pred_label, target)
        results[name] = value.item()

    return results


@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    assert logger is not None, "Need logger"
    logger = cast("MLFlowLogger", logger)

    table_path = mlflow.artifacts.download_artifacts(config.preds_uri)
    with open(table_path) as file:
        json_data = json.load(file)

    df = pd.DataFrame(json_data["data"], columns=json_data["columns"])
    results = evaluate(df, config.t)
    logger.experiment.log_table(logger.run_id, results, "slide_metrics.json")


if __name__ == "__main__":
    main()
