from typing import TypeAlias, TypedDict

from torch import Tensor


# How does one row in slides.parquet look like
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


# how does tile metadata look like in TL dataset
class TileMetadata(TypedDict):
    slide: str
    x: int
    y: int


class TileMetadataBatch(TypedDict):
    slide: list[str]
    x: Tensor
    y: Tensor


LabeledTileSample: TypeAlias = tuple[Tensor, Tensor, TileMetadata] # Image | label | Metadata
UnlabeledTileSample: TypeAlias = tuple[Tensor, TileMetadata] # Image | Metadata

LabeledTileSampleBatch: TypeAlias = tuple[Tensor, Tensor, TileMetadataBatch] # Images | labels | Metadata
UnlabeledTileSampleBatch: TypeAlias = tuple[Tensor, TileMetadataBatch] # Images | labels | Metadata


# how does slide metadata in bag dataset look like
class SlideMetadata(TypedDict):
    slide_id: str
    slide_name: str
    slide_path: str
    xs: Tensor
    ys: Tensor


LabeledBagOfTilesSample = tuple[
    Tensor, Tensor, Tensor, SlideMetadata
]  # tiles / embeddings, tl_labels, sl_labels, metadata
SLLabeledBagOfTilesSample = tuple[
    Tensor, Tensor, SlideMetadata
]  # tiles / embeddings, sl_label, metadata (no TL labels)
UnlabeledBagOfTilesSample = tuple[Tensor, SlideMetadata]  # tiles / embeddings, metadata

LabeledBagOfTilesSampleBatch: TypeAlias = tuple[
    Tensor, Tensor, Tensor, list[SlideMetadata]
]
SLLabeledBagOfTilesSampleBatch: TypeAlias = tuple[Tensor, Tensor, list[SlideMetadata]]
UnlabeledBagOfTilesSampleBatch: TypeAlias = tuple[Tensor, list[SlideMetadata]]

MILModelOutput = tuple[
    Tensor, Tensor, Tensor, Tensor
]  # SL preds, TL preds, Padding TL mask, Attention
