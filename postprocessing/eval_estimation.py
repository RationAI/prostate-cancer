from itertools import product

import hydra
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import auc, roc_curve

from postprocessing.read_table import read_json_table


def get_auc(table: pd.DataFrame, to_estimate: dict[str, list[int]]) -> dict[str, float]:
    aucs = {}
    param_names = list(to_estimate.keys())
    values_product = list(product(*to_estimate.values()))
    for configuration in values_product:
        keys = [f"{param_names[i]}={configuration[i]}" for i in range(len(to_estimate))]
        key_str = "_".join(keys)
        fpr, tpr, _ = roc_curve(table["target"], table[f"pred_{key_str}"])
        # mlflow does not allow "=" in metric name
        aucs[f"auc_{key_str.replace('=', '_')}"] = float(auc(fpr, tpr))

    return aucs


@with_cli_args(["+postprocessing=eval_estimation"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = read_json_table(config.preds_uri)
    aucs = get_auc(df, config.to_estimate)
    logger.log_metrics(aucs)
    logger.log_hyperparams({"max. AUC configuration": max(aucs, key=lambda x: aucs[x])})


if __name__ == "__main__":
    main()
