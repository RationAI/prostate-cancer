# Copyright (c) The RationAI team.

from numpy.typing import NDArray

from prostate_cancer.datamodule.augmentations.her_separation import HERSeparation


class HER2RGB(HERSeparation):
    """Image augmentation performing conversion from HER (Hematoxylin-eosin-residual) to RGB colorspace."""

    def apply(self, img: NDArray, **params) -> NDArray:
        """Applies transformation to the image.

        Args:
            img: Image in HER colorspace, final dimension denotes channels
            params: Additional parameters
        Returns:
            Image of the same shape in RGB colorspace
        """
        return self.convert_to_rgb(img)
