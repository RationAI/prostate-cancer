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


@with_cli_args(["+preprocessing=k_fold"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:

    # --- Load file with groups ---
    annotations_source = pd.read_csv(
        mlflow.artifacts.download_artifacts(config.data.metadata_table)
    )

    # --- Load tilings ---
    tiling_path_512 = (
        mlflow.artifacts.download_artifacts(config.data.tiles_uri_512)
        if config.data.tiles_uri_512
        else None
    )
    tiling_path_224 = (
        mlflow.artifacts.download_artifacts(config.data.tiles_uri_224)
        if config.data.tiles_uri_224
        else None
    )

    if not tiling_path_512 and not tiling_path_224:
        raise ValueError("At least one tiling must be present")

    slides_df_512, tiles_df_512 = None, None
    slides_df_224, tiles_df_224 = None, None

    if tiling_path_512:
        slides_df_512 = pd.read_parquet(f"{tiling_path_512}/slides.parquet")
        tiles_df_512 = pd.read_parquet(f"{tiling_path_512}/tiles.parquet")

    if tiling_path_224:
        slides_df_224 = pd.read_parquet(f"{tiling_path_224}/slides.parquet")
        tiles_df_224 = pd.read_parquet(f"{tiling_path_224}/tiles.parquet")

    # --- Choose base slides (for fold computation) ---
    base_slides_df = slides_df_512 if slides_df_512 is not None else slides_df_224
    if base_slides_df is None:
        raise ValueError("At least one DF needs to be present")

    # --- Join annotations ---
    base_slides_df = base_slides_df.join(
        annotations_source.set_index("slide_path")[config.group_column],
        on="path",
        how="left",
        validate="one_to_one",
    )

    if base_slides_df[config.group_column].isna().any():
        raise ValueError(
            f"Missing '{config.group_column}' after joining annotations source."
        )

    # --- Compute folds ---
    slides_with_folds = stratified_group_k_fold_split(
        base_slides_df,
        config.target_column,
        config.group_column,
        config.k,
    )
    fold_by_path = slides_with_folds.set_index("path")["fold"]

    group_to_folds = slides_with_folds.groupby(config.group_column)["fold"].nunique()
    leaking_groups = group_to_folds[group_to_folds > 1]
    if len(leaking_groups) != 0:
        raise ValueError("Cross-fold patient leakage")

    # --- Save datasets ---
    if slides_df_512 is not None:
        slides_df_512 = slides_df_512.join(
            fold_by_path, on="path", how="left", validate="one_to_one"
        )
        if slides_df_512["fold"].isna().any():
            raise ValueError("Missing fold assignment for some slides")

        save_mlflow_dataset(
            slides=slides_df_512,
            tiles=tiles_df_512,
            dataset_name=f"{config.data.data_name}_512_with_folds",
        )

    if slides_df_224 is not None:
        slides_df_224 = slides_df_224.join(
            fold_by_path, on="path", how="left", validate="one_to_one"
        )
        if slides_df_224["fold"].isna().any():
            raise ValueError("Missing fold assignment for some slides")

        save_mlflow_dataset(
            slides=slides_df_224,
            tiles=tiles_df_224,
            dataset_name=f"{config.data.data_name}_224_with_folds",
        )


if __name__ == "__main__":
    main()
