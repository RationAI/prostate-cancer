from typing import cast

import hydra
import ray
import torch
from omegaconf import DictConfig
from rationai.masks import process_items
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger

from prostate_cancer.data.datasets import UnlabeledTilesDataset
from prostate_cancer.data.datasets.tile_dataset import SlideTiles


@ray.remote
class StatsActor:
    """Actor to store global pixel sums and squared sums for computing mean/std."""

    def __init__(self) -> None:
        self.sum = torch.zeros(3, dtype=torch.float64)
        self.sum_sq = torch.zeros(3, dtype=torch.float64)
        self.count: int = 0

    def add_stats(self, sum_: torch.Tensor, sum_sq: torch.Tensor, count: int) -> None:
        self.sum += sum_
        self.sum_sq += sum_sq
        self.count += count

    def get_mean(self) -> torch.Tensor:
        assert self.count != 0
        return self.sum / self.count

    def get_std(self) -> torch.Tensor:
        mean = self.get_mean()
        return torch.sqrt(self.sum_sq / self.count - mean**2)


stats_actor = StatsActor.remote()  # type: ignore[attr-defined]


@ray.remote
def process_slide(slide: SlideTiles) -> None:
    """Process the slide.

    Read the slide tiles and save the mean and std of tiles into the stats actor.

    Arguments:
        slide (SlideTiles): Slide dataset.
    """
    for i in range(len(slide)):
        tile = slide[i]
        assert len(tile) == 2
        x, _ = tile
        x = x.float()  # x shape is (C, H, W)

        sum_ = x.sum(dim=(1, 2))
        sum_sq = (x**2).sum(dim=(1, 2))
        count = x.shape[1] * x.shape[2]

        stats_actor.add_stats.remote(sum_, sum_sq, count)


@hydra.main(
    config_path="../configs",
    config_name="preprocessing/mean_and_std",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    logger.experiment.log_param(logger.run_id, "dataset_uris", config.uris)
    dataset = UnlabeledTilesDataset(
        uris=config.uris,
        thresholds=config.thresholds,
    )

    slides = cast("list[SlideTiles]", dataset.datasets)
    process_items(slides, process_item=process_slide)

    mean = ray.get(stats_actor.get_mean.remote())
    std = ray.get(stats_actor.get_std.remote())

    print(f"Mean: {mean}")
    print(f"Std: {std}")

    logger.experiment.log_param(
        logger.run_id, "mean", [round(m, 4) for m in mean.tolist()]
    )
    logger.experiment.log_param(
        logger.run_id, "std", [round(s, 4) for s in std.tolist()]
    )

    print("Successfully logged mean and std to MLflow")


if __name__ == "__main__":
    main()
