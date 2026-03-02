import os
import tempfile

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from postprocessing.read_table import read_json_table


@with_cli_args(["+postprocessing=slide_prediction_table_merge"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = None
    for model, table_uri in config.table_uris.items():
        path = mlflow.artifacts.download_artifacts(table_uri)
        if path.endswith("json"):
            current = read_json_table(path)
        else:
            current = pd.read_csv(mlflow.artifacts.download_artifacts(table_uri))

        if df is None:  # first iteration
            df = current

            # may not be present if not ehanced table
            if "is_fp" in df.columns and "bin_prediction" in df.columns:
                df = df.drop(["is_fp", config.pred_column, "bin_prediction"], axis=1)
            else:
                df = df.drop([config.pred_column], axis=1)

        df[f"{model}_pred_score"] = current[config.pred_column]

        # may not be present if not ehanced table
        if "bin_prediction" in current.columns:
            df[f"{model}_pred_binary"] = current["bin_prediction"]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "sl_preds_merged.csv")
        if df is not None:
            df.to_csv(out_path, index=False)
            logger.log_artifact(out_path, artifact_path="tables")


if __name__ == "__main__":
    main()
