# Copyright (c) The RationAI team.

from numpy.typing import NDArray
from skimage.color import separate_stains

from prostate_cancer.datamodule.augmentations.her_separation import HERSeparation


class RGB2HER(HERSeparation):
    """Image augmentation performing conversion from RGB to HER (Hematoxylin-eosin-residual) colorspace."""

    def apply(self, img: NDArray, **params) -> NDArray:
        """Applies transformation to the image.

        Args:
            img: Image in RGB colorspace, final dimension denotes channels
            params: Additional parameters
        Returns:
            Image of the same shape in HER colorspace
        """
        return separate_stains(rgb=img, conv_matrix=self.her_from_rgb)
