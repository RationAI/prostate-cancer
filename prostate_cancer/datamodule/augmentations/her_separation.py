# Copyright (c) The RationAI team.

import albumentations
import numpy as np
from numpy.typing import NDArray
from skimage.color import combine_stains


class HERSeparation(albumentations.ImageOnlyTransform):
    """Custom image augmentation performing HER (Hematoxylin-eosin-residual) separation from HE images."""

    def __init__(
        self,
        h_vector: list[float],
        e_vector: list[float],
        r_vector: list[float],
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p, always_apply=always_apply)
        self.rgb_from_her = np.array([h_vector, e_vector, r_vector])
        self.her_from_rgb = np.linalg.inv(self.rgb_from_her)

    def convert_to_rgb(self, im: NDArray) -> NDArray:
        rgb_img = combine_stains(stains=im, conv_matrix=self.rgb_from_her)
        return (rgb_img * 255).astype(np.uint8)

    def apply(self, img: NDArray, **params) -> NDArray:
        raise NotImplementedError
