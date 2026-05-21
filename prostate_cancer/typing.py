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


LabeledSlideSample = tuple[Tensor, Tensor, Tensor, SlideMetadata]
UnlabeledSlideSample = tuple[Tensor, SlideMetadata]

LabeledSlideSampleBatch: TypeAlias = tuple[Tensor, Tensor, Tensor, list[SlideMetadata]]
UnlabeledSlideSampleBatch: TypeAlias = tuple[Tensor, list[SlideMetadata]]

MILModelOutput = tuple[Tensor, Tensor, Tensor]  # SL preds, TL preds, Padding TL mask
