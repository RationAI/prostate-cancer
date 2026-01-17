import os
import tempfile

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


@hydra.main(
    config_path="../configs",
    config_name="postprocessing/slide_predictions_table_merge",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = None
    for model, table_uri in config.table_uris.items():
        current = pd.read_csv(mlflow.artifacts.download_artifacts(table_uri))
        if df is None:  # first iteration
            df = current
            df = df.drop(["is_fp", "prediction", "bin_prediction"], axis=1)

        df[f"{model}_pred_score"] = current["prediction"]
        df[f"{model}_pred_binary"] = current["bin_prediction"]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "sl_preds_merged.csv")
        if df is not None:
            df.to_csv(out_path, index=False)
            logger.log_artifact(out_path, artifact_path="tables")


if __name__ == "__main__":
    main()
