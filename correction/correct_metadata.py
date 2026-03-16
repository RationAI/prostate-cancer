"""Metadata Correction.

We have found a bug in MMCI 2k exploration script, since this bug affects carcinoma labels,
a new data split would differ from the current one (experimentally confirmed).
For that reason, we add script to correct metadata tables.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import with_cli_args
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


@with_cli_args(["+correction=correct_metadata"])
@hydra.main(config_path="../configs", config_name="correction", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    sot_df = pd.read_csv(mlflow.artifacts.download_artifacts(config.source_of_truth))
    meta_path = mlflow.artifacts.download_artifacts(config.metadata_to_correct)
    meta_csv_df = pd.read_csv(meta_path)
    cols = list(meta_csv_df.columns)
    meta_csv_df = meta_csv_df.merge(
        sot_df[["slide_path", "carcinoma"]], on="slide_path", how="left"
    )

    meta_csv_df["carcinoma"] = meta_csv_df["carcinoma_y"]
    meta_csv_df = meta_csv_df[cols]

    with TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / f"{Path(meta_path).stem}_corrected.csv"
        meta_csv_df.to_csv(str(target), index=False)
        mlflow.log_artifact(str(target))


if __name__ == "__main__":
    main()
