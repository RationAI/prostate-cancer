import tempfile
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


@hydra.main(
    config_path="../configs",
    config_name="exploration/comenius_3_sample",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = Path(config.data_dir).glob("*.mrxs")
    explored = pd.DataFrame({"slide_path": list(slides)})

    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / "comenius_3_sample_prostate.csv"
        explored.to_csv(target, index=False)
        logger.log_artifact(str(target))


if __name__ == "__main__":
    main()
