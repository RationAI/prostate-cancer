from typing import TypeAlias, TypedDict

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
    slide_path: str
    xs: Tensor
    ys: Tensor


class SlideMetadataBatch(TypedDict):
    slide_id: list[str]
    slide_name: list[str]
    slide_path: list[str]
    n_tiles: list[int]
    xs: Tensor
    ys: Tensor


LabeledSlideSample = tuple[Tensor, Tensor, TileMetadata]
UnlabeledSlideSample = tuple[Tensor, TileMetadata]

LabeledSlideSampleBatch: TypeAlias = tuple[Tensor, Tensor, SlideMetadataBatch]
UnlabeledSlideSampleBatch: TypeAlias = tuple[Tensor, SlideMetadataBatch]

MILModelOutput = tuple[Tensor, Tensor]  # SL preds, TL preds
