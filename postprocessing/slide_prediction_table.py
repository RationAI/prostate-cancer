import os
import tempfile

import hydra
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger

from postprocessing.read_table import read_json_table


@hydra.main(
    config_path="../configs",
    config_name="postprocessing/slide_predictions_table",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = read_json_table(config.preds_uri)
    df["bin_prediction"] = df["prediction"] >= config.t
    df["is_fp"] = df["bin_prediction"] & ~df["target"]
    df = df.sort_values(by="prediction", ascending=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "sl_preds.csv")
        df.to_csv(out_path, index=False)
        logger.log_artifact(out_path, artifact_path="tables")


if __name__ == "__main__":
    main()
