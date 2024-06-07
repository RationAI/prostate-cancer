from statistics import mean

from rationai.empaia.typing import slide


def find_closes_resolution_level(
    levels: list[slide.SlideLevel],
    pixel_size_nm: slide.SlidePixelSizeNm,
    target_resolution: float,
) -> int:
    """Find the closest level that matches the target resolution.

    Args:
        levels: The levels of the whole slide image.
        pixel_size_nm: The pixel size of the whole slide image.
        target_resolution: The target resolution in μm/px.

    Returns:
        The index of the level that matches the target resolution.
    """
    base_resolution = mean((pixel_size_nm.x, pixel_size_nm.y)) / 1000  # μm/px
    resolutions = [base_resolution * level.downsample_factor for level in levels]
    return min(enumerate(resolutions), key=lambda x: abs(x[1] - target_resolution))[0]
