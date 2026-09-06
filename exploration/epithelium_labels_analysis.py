"""Numerical error estimation of using epithelium based labels instead of manual annotations."""

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


@hydra.main(
    config_path="../configs",
    config_name="exploration/colorectum/epithelium_labels_analysis",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tiling = mlflow.artifacts.download_artifacts(config.data.tiles_filtered_uri_224)
    tiles = pd.read_parquet(tiling + "/tiles.parquet")
    carcinoma_tiles = (
        tiles["carcinoma_roi_percentage"] > config.data.thresholds.carcinoma_roi_t
    )

    # this works because all slides in the dataset are positive
    epithelium_tiles = (
        tiles["epithelium_roi_percentage"] > config.data.thresholds.epithelium_roi_t
    )

    cm = confusion_matrix(carcinoma_tiles, epithelium_tiles)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": accuracy_score(carcinoma_tiles, epithelium_tiles),
        "precision": precision_score(carcinoma_tiles, epithelium_tiles),
        "recall": recall_score(carcinoma_tiles, epithelium_tiles),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }
    mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
