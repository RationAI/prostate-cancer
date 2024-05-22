from typing import Literal, Type, TypedDict


class Color(TypedDict):
    r: int
    g: int
    b: int
    a: int


class Channel(TypedDict):
    id: int
    name: str
    color: Color


class Extent(TypedDict):
    x: int
    y: int
    z: int


class Level(TypedDict):
    extent: Extent
    downsample_factor: float


class PixelSize(TypedDict):
    x: float
    y: float
    z: float | None


class WSI(TypedDict):
    id: str
    channels: list[Channel]
    channel_depth: int
    extent: Extent
    num_levels: int
    pixel_size_nm: PixelSize
    tile_extent: Extent
    levels: list[Level]
    format: str
    raw_download: bool
    tissue: str
    stain: str


class ChannelClass(TypedDict):
    number_value: int
    class_value: str


class PixelmapLevel(TypedDict):
    slide_level: int
    position_min_x: int
    position_min_y: int
    position_max_x: int
    position_max_y: int


class ContinuousPixelmap(TypedDict):
    id: str
    name: str
    reference_id: str
    reference_type: str
    creator_id: str
    creator_type: str
    type: Literal["continuous_pixelmap"]
    element_type: Literal["float32", "float64"]
    min_value: float
    neutral_value: float
    max_value: float
    tilesize: int
    channel_count: int
    channel_class_mapping: list[ChannelClass]
    levels: list[PixelmapLevel]


class WSIMask(TypedDict):
    id: str
    local_id: str
