from collections.abc import Sequence

from datasets import Dataset as HFDataset
from torch.utils.data import WeightedRandomSampler

from prostate_cancer.datamodule.datasets import LabeledTilesDataset


class TileStratifiedWeightedRandomSampler(WeightedRandomSampler):
    def __init__(
        self,
        dataset: LabeledTilesDataset,
        target_col: str,
        replacement: bool = True,
    ) -> None:

        weights = self._get_weights(dataset.tiles, target_col)

        super().__init__(
            weights,
            num_samples=len(dataset),
            replacement=replacement,
        )

    def _get_weights(self, ds: HFDataset, target_col: str) -> Sequence[float]:
        labels = ds[target_col]

        counts = {}
        for v in labels:
            counts[v] = counts.get(v, 0) + 1

        weights = [1.0 / counts[v] for v in labels]
        return weights
