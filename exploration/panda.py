import tempfile
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


@hydra.main(
    config_path="../configs",
    config_name="exploration/panda",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    metadata = pd.read_csv(mlflow.artifacts.download_artifacts(config.full_metadata_csv_uri))

    mask = (metadata["annotation"] == True) & (metadata["is_annotation_corrupted"] == False) & (metadata["is_wsi_valid"] == True)
    metadata = metadata[ mask ]
    metadata = metadata.rename(columns={"is_carcinoma": "carcinoma"})

    radboud_metadata = metadata[ metadata["data_provider"] == "radboud" ]
    karolinska_metadata = metadata[ metadata["data_provider"] == "karolinska" ]

    with tempfile.TemporaryDirectory() as temp_dir:
        radboud_target = Path(temp_dir) / "radboud_metadata.csv"
        radboud_metadata.to_csv(Path(temp_dir) / "radboud_metadata.csv", index=False)
        logger.log_artifact(str(radboud_target))

        karolinska_target = Path(temp_dir) / "karolinska_metadata.csv"
        karolinska_metadata.to_csv(Path(temp_dir) / "karolinska_metadata.csv", index=False)
        logger.log_artifact(str(karolinska_target))

if __name__ == "__main__":
    main()
