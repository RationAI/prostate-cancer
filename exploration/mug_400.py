import tempfile
from pathlib import Path

import hydra
import pandas as pd
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


@hydra.main(
    config_path="../configs",
    config_name="exploration/mug_400",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    client = MlflowClient()
    run = client.get_run(config.sl_annotations_run_id)
    annotations = run.data.metrics  # SL metrics were stored as metrics
    annotations = {key.split("_")[-1]: val for key, val in annotations.items()}

    slides = Path(config.data_dir).glob("*.svs")
    slides_filtered = []
    sl_carcinoma = []
    for slide in slides:
        if (
            slide.name not in config.blacklist
        ):  # not HE stained, or otherwise corrupted slides
            slides_filtered.append(slide)
            assert slide.name in annotations
            sl_carcinoma.append(annotations[slide.name] == 1)

    explored = pd.DataFrame({"slide_path": slides_filtered, "carcinoma": sl_carcinoma})

    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / "mug_400.csv"
        explored.to_csv(target, index=False)
        logger.log_artifact(str(target))


if __name__ == "__main__":
    main()
