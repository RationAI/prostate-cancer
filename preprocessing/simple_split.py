from pathlib import Path
import tempfile

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.model_selection import train_test_split


@with_cli_args(["+preprocessing=simple_split"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides_df_path = mlflow.artifacts.download_artifacts(config.data.metadata_table)
    slides_df = pd.read_csv(slides_df_path)

    train_slides, test_slides = train_test_split(
        slides_df,
        test_size=config.test_size,
        random_state=42,
        stratify=slides_df[config.target_column],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        train_out = Path(tmpdir) / f"{config.data.data_name}_train.csv"
        train_slides.to_csv(train_out, index=False)
        logger.log_artifact(str(train_out))

        test_out = Path(tmpdir) / f"{config.data.data_name}_test.csv"
        test_slides.to_csv(test_out, index=False)
        logger.log_artifact(str(test_out))


if __name__ == "__main__":
    main()
