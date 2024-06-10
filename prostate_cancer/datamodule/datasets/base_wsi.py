# Copyright (c) The RationAI team.

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from prostate_cancer.datamodule.samplers import BaseSampler


class BaseDataset(
    torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor, Any | None]]
):
    sampler: BaseSampler
    _epoch_samples: list

    def __init__(self, sampler: BaseSampler, seed: int) -> None:
        self._epoch_samples = []
        self.sampler = sampler
        self._rng = np.random.default_rng(seed)

    def generate_samples(self) -> None:
        self._epoch_samples = self.sampler.get_sample()

    def __len__(self) -> int:
        return len(self._epoch_samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, Any | None]:
        raise NotImplementedError


def extract_tile(
    slide_fp: Path, coord_x: int, coord_y: int, tile_size: int, level: int
) -> NDArray[np.uint8]:
    """Extracts a tile from a slide using the supplied coordinate values.

    Args:
        slide_fp (Path): Path to the slide.
        coord_x (int): Coordinates of a tile to be extracted at OpenSlide level 0 resolution.
        coord_y (int): Coordinates of a tile to be extracted at OpenSlide level 0 resolution.
        tile_size (int): Size of the tile to be extracted.
        level (int): Resolution level from which tile should be extracted.

    Returns:
        NDArray: RGB Tile represented as numpy array.
    """
    raise NotImplementedError
