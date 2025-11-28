import hydra
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import auc, roc_curve

from postprocessing.read_table import read_json_table


def get_auc(table: pd.DataFrame, kernel_sizes: list[int]) -> dict[str, float]:
    aucs = {}
    for ks in kernel_sizes:
        fpr, tpr, _ = roc_curve(table["target"], table[f"pred_{ks}"])
        aucs[f"auc_kernel_size_{ks}"] = float(auc(fpr, tpr))

    return aucs


@hydra.main(
    config_path="../configs",
    config_name="postprocessing/kernel_estimation",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = read_json_table(config.preds_uri)
    aucs = get_auc(df, config.kernel_sizes)
    logger.log_metrics(aucs)
    logger.log_hyperparams({"max. auc kernel size": max(aucs, key=lambda x: aucs[x])})


if __name__ == "__main__":
    main()
