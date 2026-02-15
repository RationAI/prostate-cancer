from pathlib import Path

import hydra
import mlflow
import pandas as pd
from mlkit.rationai.mlkit import with_cli_args
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.model_selection import StratifiedGroupKFold


# Find the split where the distribution of classes is the closest to the original distribution
# This is done because the size of the test dataset is not exactly the same as the parameter `test_size`
def get_distribution(labels: pd.Series) -> pd.Series:
    return labels.value_counts(normalize=True).sort_index()


def stratified_group_split(
    original_data: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into train and test sets using StratifiedGroupKFold.

    This function ensures that the split is stratified based on the labels (carcinoma) and
    grouped by the groups (cases in our case). The split is done in such a way that the distribution
    of the labels in the test set is as close as possible to the distribution in
    the original data. The function also ensures that each class is present
    in the test set if possible.

    Note:
    The underlying functionality is originally used for cross-validation folds generation, however
    since train_test_split from sklearn does not support both groups and stratified split at once
    we perfrom train-test split by selecting the "best" - (by label distribution) fold we can get.
    Moreover, a simple train-test split on the level of cases and then just collecting matching slides
    does not guarantee to preserve label distribution.

    Arguments:
        original_data (pd.DataFrame): DataFrame with the data to split.
        labels (pd.Series): Series with the label information, for stratification.
        groups (pd.Series): Series with the group information, for grouping.
        test_size (float): Proportion of the test set. Default is 0.2.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and validation splits.
    """
    data = original_data.copy()  # To avoid modifying the original data

    assert len(data) == len(labels) and len(data) == len(groups), (
        "The number of rows in data, labels, and groups must be the same."
    )
    assert 0 < test_size < 1, "test_size must be in the interval (0, 1)."

    n_splits = round(1 / test_size)
    if abs((1 / n_splits) - test_size) > 0.05:
        print(
            f"Warning: actual test fraction ≈ {1 / n_splits:.2f}, not {test_size:.2f}"
        )

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    data_distribution = get_distribution(labels)
    min_diff: float | None = None
    train_idx: list[int] | None = None
    test_idx: list[int] | None = None

    for curr_train_idx, curr_test_idx in sgkf.split(X=data, y=labels, groups=groups):
        test_distribution = get_distribution(labels.iloc[curr_test_idx])
        diff = (test_distribution - data_distribution).abs().sum()

        # Ensure that each class is present in the test set
        if len(test_distribution) != len(data_distribution):
            diff = float("inf")

        if min_diff is None or diff < min_diff:
            min_diff = diff
            train_idx = curr_train_idx
            test_idx = curr_test_idx

    assert train_idx is not None and test_idx is not None

    train_data = original_data.iloc[train_idx]
    test_data = original_data.iloc[test_idx]

    return train_data, test_data


@with_cli_args(["+preprocessing=data_split"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides_df_path = mlflow.artifacts.download_artifacts(config.slides_df_uri)
    slides_df = pd.read_csv(slides_df_path)

    train_slides, val_slides = stratified_group_split(
        original_data=slides_df,
        labels=slides_df[config.target_column],
        groups=slides_df["case_id"],
        test_size=config.test_size,
        random_state=42,
    )

    print("Train slides:", train_slides[config.target_column].value_counts())
    print("Validation slides:", val_slides[config.target_column].value_counts())

    train_out = Path("train_slides.csv")
    train_slides.to_csv(str(train_out), index=False)
    logger.log_artifact("train_slides.csv")
    train_out.unlink()

    val_out = Path("val_slides.csv")
    val_slides.to_csv(str(val_out), index=False)
    logger.log_artifact("val_slides.csv")
    val_out.unlink()


if __name__ == "__main__":
    main()
