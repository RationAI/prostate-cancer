import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset
from sklearn.model_selection import StratifiedGroupKFold


def stratified_group_k_fold_split(
    slides_df: pd.DataFrame,
    target_col: str,
    group_col: str,
    k: int,
) -> pd.DataFrame:
    slides_df = slides_df.copy()
    fold_mask = np.empty(len(slides_df), dtype=int)

    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    folds = sgkf.split(
        slides_df,
        slides_df[target_col],
        slides_df[group_col],
    )

    for i, (_, fold_index) in enumerate(folds):
        fold_mask[fold_index] = i

    slides_df["fold"] = fold_mask
    return slides_df


def load_tiling_tables(uri: str) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    tiling_path = mlflow.artifacts.download_artifacts(uri) if uri else None
    if tiling_path is None:
        return None, None

    slides_df = pd.read_parquet(f"{tiling_path}/slides.parquet")
    tiles_df = pd.read_parquet(f"{tiling_path}/tiles.parquet")
    return slides_df, tiles_df


def attach_group(base_slides_df: pd.DataFrame, annotations_source: pd.DataFrame, group_col: str) -> pd.DataFrame:
    base_slides_df = base_slides_df.join(
        annotations_source.set_index("slide_path")[group_col],
        on="path",
        how="left",
        validate="one_to_one",
    )

    if base_slides_df[group_col].isna().any():
        raise ValueError(
            f"Missing '{group_col}' after joining annotations source."
    )

    return base_slides_df


def compute_folds(base_slides_df: pd.DataFrame, target_column: str, group_column: str, k: int) -> pd.DataFrame:
    slides_with_folds = stratified_group_k_fold_split(base_slides_df, target_column, group_column, k)
    fold_by_path = slides_with_folds.set_index("path")["fold"]

    group_to_folds = slides_with_folds.groupby(group_column)["fold"].nunique()
    leaking_groups = group_to_folds[group_to_folds > 1]
    if len(leaking_groups) != 0:
        raise ValueError("Cross-fold patient leakage")
    
    return fold_by_path


def apply_folds(slides_df: pd.DataFrame, tiles_df: pd.DataFrame, fold_by_path: pd.DataFrame, name: str) -> None:
    slides_df = slides_df.join(
        fold_by_path, on="path", how="left", validate="one_to_one"
    )
    if slides_df["fold"].isna().any():
        raise ValueError("Missing fold assignment for some slides")

    save_mlflow_dataset(
        slides=slides_df,
        tiles=tiles_df,
        dataset_name=f"{name}_with_folds",
    )


@with_cli_args(["+preprocessing=k_fold"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    annotations_source = pd.read_csv(
        mlflow.artifacts.download_artifacts(config.data.metadata_table)
    )

    slides_df_512, tiles_df_512 = load_tiling_tables(config.data.tiles_uri_512)
    slides_df_224, tiles_df_224 = load_tiling_tables(config.data.tiles_uri_224)

    if slides_df_512 is None and slides_df_224 is None:
        raise ValueError("At least one tiling must be present")

    base_slides_df = slides_df_512 if slides_df_512 is not None else slides_df_224

    base_slides_df = attach_group(base_slides_df, annotations_source, config.group_column)
    fold_by_path = compute_folds(base_slides_df, config.target_column, config.group_column, config.k)

    if slides_df_512 is not None:
        apply_folds(slides_df_512, tiles_df_512, fold_by_path, f"{config.data.data_name}_512")

    if slides_df_224 is not None:
        apply_folds(slides_df_224, tiles_df_224, fold_by_path, f"{config.data.data_name}_224")


if __name__ == "__main__":
    main()
