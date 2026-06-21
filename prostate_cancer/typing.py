from typing import TypeAlias, TypedDict

from torch import Tensor


class TilingSlideMetadata(TypedDict):
    id: str
    path: str
    extent_x: int
    extent_y: int
    tile_extent_x: int
    tile_extent_y: int
    stride_x: int
    stride_y: int
    mpp_x: float
    mpp_y: float
    level: int
    downsample: float
    carcinoma: bool


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


LabeledBagOfTilesSample = tuple[
    Tensor, Tensor, Tensor, SlideMetadata
]  # tiles / embeddings, tl_labels, sl_labels, metadata
UnlabeledBagOfTilesSample = tuple[Tensor, SlideMetadata]  # tiles / embeddings, metadata

LabeledBagOfTilesSampleBatch: TypeAlias = tuple[
    Tensor, Tensor, Tensor, list[SlideMetadata]
]
UnlabeledBagOfTilesSampleBatch: TypeAlias = tuple[Tensor, list[SlideMetadata]]

MILModelOutput = tuple[
    Tensor, Tensor, Tensor, Tensor
]  # SL preds, TL preds, Padding TL mask, Attention
