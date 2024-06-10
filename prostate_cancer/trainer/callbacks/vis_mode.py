# Copyright (c) The RationAI team.

from numpy.typing import NDArray


class VisualisationMode:
    zero_offset: float

    @classmethod
    def scale(cls, x: NDArray) -> NDArray:
        return (x * 255) - cls.zero_offset


class BipolarMode(VisualisationMode):
    zero_offset = 127.5

    @classmethod
    def scale(cls, x: NDArray) -> NDArray:
        return super().scale(1 - x)


class IdentityMode(VisualisationMode):
    zero_offset = 0.0

    @classmethod
    def scale(cls, x: NDArray) -> NDArray:
        return super().scale(x)
