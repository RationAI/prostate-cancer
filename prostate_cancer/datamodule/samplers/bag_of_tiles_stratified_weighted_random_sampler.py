from collections.abc import Sequence

from datasets import Dataset as HFDataset
from torch.utils.data import WeightedRandomSampler

from prostate_cancer.datamodule.datasets import LabeledBagOfEmbeddingsDataset


class BagOfTilesStratifiedWeightedRandomSampler(WeightedRandomSampler):
    def __init__(
        self,
        dataset: LabeledBagOfEmbeddingsDataset,
        target_col: str,
        replacement: bool = True,
    ) -> None:
        super().__init__(
            self._get_weights(dataset.slides, target_col),
            num_samples=len(dataset),
            replacement=replacement,
        )

    def _get_weights(self, ds: HFDataset, target_col: str) -> Sequence[float]:
        labels = ds[target_col]

        counts: dict[bool, int] = {}
        for v in labels:
            counts[v] = counts.get(v, 0) + 1

        weights = [1.0 / counts[v] for v in labels]
        return weights
