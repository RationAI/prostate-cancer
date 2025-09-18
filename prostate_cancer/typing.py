from typing import TypeAlias, TypedDict

from torch import Tensor


class Metadata(TypedDict):
    slide: str
    x: int
    y: int


class MetadataBatch(TypedDict):
    slide: list[str]
    x: Tensor
    y: Tensor


LabeledSample = tuple[Tensor, Tensor, Metadata]
UnlabeledSample = tuple[Tensor, Metadata]

LabeledSampleBatch: TypeAlias = tuple[Tensor, Tensor, MetadataBatch]
UnlabeledSampleBatch: TypeAlias = tuple[Tensor, MetadataBatch]
