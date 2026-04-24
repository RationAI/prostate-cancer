from collections.abc import Sequence

import pandas as pd
from torch.utils.data import WeightedRandomSampler

from prostate_cancer.datamodule.datasets import LabeledTilesDataset


class TileStratifiedWeightedRandomSampler(WeightedRandomSampler):
    def __init__(
        self, dataset: LabeledTilesDataset, target_col: str, replacement: bool = True
    ) -> None:
        super().__init__(
            self._get_weights(dataset.tiles, target_col),
            num_samples=len(dataset),
            replacement=replacement,
        )

    def _get_weights(self, df: pd.DataFrame, target_col: str) -> Sequence[float]:
        value_counts = df[target_col].value_counts()
        weights = 1 / df[target_col].map(value_counts)
        return weights.tolist()
