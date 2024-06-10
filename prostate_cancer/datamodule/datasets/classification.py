# Copyright (c) The RationAI team.

import random
from pathlib import Path

import albumentations
import torch

from prostate_cancer.datamodule.datasets.base_wsi import BaseDataset, extract_tile
from prostate_cancer.datamodule.samplers import BaseSampler


class ClassificationDataset(BaseDataset):
    transforms: albumentations.TemplateTransform | None

    def __init__(
        self,
        sampler: BaseSampler,
        seed: int,
        augmentations: albumentations.TemplateTransform | None = None,
    ) -> None:
        super().__init__(sampler=sampler, seed=seed)
        self.transforms = augmentations

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        sample = self._epoch_samples[index]

        image = extract_tile(
            slide_fp=Path(sample.get("slide_fp")).resolve(),
            coord_x=sample["coord_x"],
            coord_y=sample["coord_y"],
            tile_size=sample["tile_size"],
            level=sample["sample_level"],
        )

        if self.transforms:
            random.seed(int(self._rng.integers(0, 2**63 - 1)))
            image = self.transforms(image=image)["image"]

        # permute to (channels, height, width)
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.FloatTensor([sample["class_id"]])
        return image, label, sample
