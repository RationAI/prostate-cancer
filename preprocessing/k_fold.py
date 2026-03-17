import json
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.model_selection import StratifiedGroupKFold


def get_slide_stats(df: pd.DataFrame, target_col: str) -> dict[str, dict[bool, int]]:
    counts = df[target_col].value_counts().sort_index()
    dist = counts / counts.sum()
    return {"counts": counts.to_dict(), "distribution": dist.to_dict()}


def get_case_stats(df: pd.DataFrame, target_col: str) -> dict[str, dict[bool, int]]:
    case_df = df.drop_duplicates("case_id")
    counts = case_df[target_col].value_counts().sort_index()
    dist = counts / counts.sum()
    return {"counts": counts.to_dict(), "distribution": dist.to_dict()}


@with_cli_args(["+preprocessing=k_fold"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slide_metadata = pd.read_csv( mlflow.artifacts.download_artifacts(config.data.metadata_table) )
    slide_metadata["fold"] = 0

    # K-fold split
    cv_sgkf = StratifiedGroupKFold(n_splits=config.k, shuffle=True, random_state=42)

    for fold_idx, (_, val_idx) in enumerate(
        cv_sgkf.split(slide_metadata, slide_metadata[config.target_column], groups=slide_metadata["case_id"]), 1
    ):
        slide_metadata.loc[val_idx, "fold"] = fold_idx

    # Cross-Fold Leakage
    for f1 in range(1, config.k + 1):
        for f2 in range(f1 + 1, config.k + 1):
            f1_cases = set(slide_metadata[slide_metadata["fold"] == f1]["case_id"])
            f2_cases = set(slide_metadata[slide_metadata["fold"] == f2]["case_id"])
            fold_overlap = f1_cases & f2_cases
            assert not fold_overlap, (
                f"Leakage between Fold {f1} and {f2}! Cases: {fold_overlap}"
            )

    # Comprehensive Reporting
    report = {
        "summary": {
            "total_slides": len(slide_metadata),
            "total_cases": int(slide_metadata["case_id"].nunique()),
            "slides": get_slide_stats(slide_metadata, config.target_column),
            "cases": get_case_stats(slide_metadata, config.target_column),
        },
        "folds": {
            f"fold_{i}": {
                "slides": get_slide_stats(slide_metadata[slide_metadata["fold"] == i], config.target_column),
                "cases": get_case_stats(slide_metadata[slide_metadata["fold"] == i], config.target_column),
            }
            for i in range(1, config.k + 1)
        },
    }

    # Artifact Logging
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        with open(tmp_path / "k_fold_report.json", "w") as f:
            json.dump(report, f, indent=4)

        slide_metadata[ ["slide_path", "case_id", "fold"] ].to_csv(tmp_path / f"{config.data.data_name}_{config.k}_folds.csv", index=False)

        logger.log_artifacts(local_dir=str(tmp_path), artifact_path="tables")


if __name__ == "__main__":
    main()
