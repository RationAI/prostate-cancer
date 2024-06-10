# Copyright (c) The RationAI team.

from collections.abc import Sequence

import numpy as np
from albumentations import ImageOnlyTransform
from numpy.typing import NDArray


class AdditiveNoise(ImageOnlyTransform):
    """Add number of array of numbers to image.

    Attributes:
        low: Lower bound of the random value that will be added to the image.
        high: Upper bound of the random value that will be added to the image.
        axis: Axis or axes along which unique noise will be applied. If None, noise
            will be applied element-wise.
        always_apply: Indicates whether this transformation should be always applied.
        p: Probability of applying the transformation.
    """

    def __init__(
        self,
        low: float = -0.005,
        high: float = 0.005,
        axis: tuple[int, ...] | int | None = None,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(always_apply, p)
        self.low = low
        self.high = high
        self.axis = (axis,) if isinstance(axis, int) else axis

    def _shape_from_axis(self, shape: tuple[int, ...]) -> Sequence[int]:
        if self.axis is None:
            return shape

        new_shape = [1] * len(shape)
        for axis in self.axis:
            new_shape[axis] = shape[axis]

        return new_shape

    def apply(
        self, img: NDArray[np.uint8 | np.float_], **params
    ) -> NDArray[np.float64]:
        """Applies transformation to the image.

        Args:
            img: Image to which the transformation will be applied.
            params: Additional parameters

        Returns:
            Image with added noise.
        """
        noise = np.random.uniform(self.low, self.high, self._shape_from_axis(img.shape))

        if img.dtype == np.uint8:
            return np.clip(img + noise, 0, 255)

        return img + noise
