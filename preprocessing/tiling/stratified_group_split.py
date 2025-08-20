import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# Find the split where the distribution of classes is the closest to the original distribution
# This is done because the size of the test dataset is not exactly the same as the parameter `test_size`
def get_distribution(labels: pd.Series) -> pd.Series:
    return labels.value_counts(normalize=True).sort_index()


def stratified_group_split(
    data: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into train and test sets using StratifiedGroupKFold.

    This function ensures that the split is stratified based on the labels and
    grouped by the groups. The split is done in such a way that the distribution
    of the labels in the test set is as close as possible to the distribution in
    the original data. The function also ensures that each class is present
    in the test set if possible.

    Arguments:
        data (pd.DataFrame): DataFrame with the data to split.
        labels (pd.Series): Series with the label information, for stratification.
        groups (pd.Series): Series with the group information, for grouping.
        test_size (float): Proportion of the test set. Default is 0.2.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        tuple[list[Path], list[Path]]: A tuple containing the train and validation splits.
    """
    data = data.copy()  # To avoid modifying the original data

    assert len(data) == len(labels) and len(data) == len(groups), (
        "The number of rows in data, labels, and groups must be the same."
    )
    assert 0 < test_size < 1, "test_size must be in the interval (0, 1)."

    sgkf = StratifiedGroupKFold(
        n_splits=int(1 / test_size), shuffle=True, random_state=random_state
    )

    data_distribution = get_distribution(data)
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

    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    return train_data, test_data
