"""Tiles Correction.

We have found a bug in MMCI 2k exploration script, since this bug affects carcinoma labels,
a new data split would differ from the current one (experimentally confirmed).
For that reason, we add scripts to correct tiles datasets.
"""

from pathlib import Path

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import with_cli_args
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset


@with_cli_args(["+correction=correct_tiles"])
@hydra.main(config_path="../configs", config_name="correction", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    sot_df = pd.read_csv(mlflow.artifacts.download_artifacts(config.source_of_truth))
    sot_df["stem"] = sot_df["slide_path"].map(lambda x: Path(x).stem)

    tiles_path = Path(mlflow.artifacts.download_artifacts(config.tiles_to_correct))
    slides = pd.read_parquet(tiles_path / "slides.parquet")
    slides["stem"] = slides["path"].map(lambda x: Path(x).stem)
    result = pd.merge(slides, sot_df, on="stem", how="inner")
    result["carcinoma"] = result["carcinoma_y"]
    final = result[slides.columns]
    tiles = pd.read_parquet(tiles_path / "tiles.parquet")
    save_mlflow_dataset(
        slides=final,
        tiles=tiles,
        dataset_name=config.data.data_name,
    )


if __name__ == "__main__":
    main()
