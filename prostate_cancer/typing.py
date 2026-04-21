from typing import TypeAlias, TypedDict
from pathlib import Path

from torch import Tensor


class TileMetadata(TypedDict):
    slide: str
    x: int
    y: int


class TileMetadataBatch(TypedDict):
    slide: list[str]
    x: Tensor
    y: Tensor


LabeledTileSample: TypeAlias = tuple[Tensor, Tensor, TileMetadata]
UnlabeledTileSample: TypeAlias = tuple[Tensor, TileMetadata]

LabeledTileSampleBatch: TypeAlias = tuple[Tensor, Tensor, TileMetadataBatch]
UnlabeledTileSampleBatch: TypeAlias = tuple[Tensor, TileMetadataBatch]


class SlideMetadata(TypedDict):
    slide_id: str
    slide_name: str
    slide_path: Path


class SlideMetadataBatch(TypedDict):
    slide_id: list[str]
    slide_name: list[str]
    slide_path: list[Path]


LabeledSlideSample = tuple[Tensor, Tensor, TileMetadata]
UnlabeledSlideSample = tuple[Tensor, TileMetadata]

LabeledSlideSampleBatch: TypeAlias = tuple[Tensor, Tensor, TileMetadataBatch]
UnlabeledSlideSampleBatch: TypeAlias = tuple[Tensor, TileMetadataBatch]
