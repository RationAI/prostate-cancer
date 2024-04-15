import asyncio
from typing import Any, Iterator

import aiohttp
import numpy as np
from numpy.typing import NDArray
from skimage.transform import downscale_local_mean


def bbox(array: NDArray[Any]) -> tuple[int, int, int, int]:
    """Get the bounding box of a binary image.

    Returns:
        cmin, rmin, cmax, rmax
    """
    rows = np.any(array, axis=1)
    cols = np.any(array, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(cmin), int(rmin), int(cmax), int(rmax)


def array_to_bulk_bytes(array: NDArray[Any], tile_size: int) -> bytes:
    array = np.pad(
        array,
        (
            (0, -array.shape[0] % tile_size),
            (0, -array.shape[1] % tile_size),
        ),
        "constant",
        constant_values=0,
    )

    return b"".join(
        array[y : y + tile_size, x : x + tile_size].tobytes()
        for y in range(0, array.shape[0], tile_size)
        for x in range(0, array.shape[1], tile_size)
    )


def create_pyramid(
    image: NDArray[Any], level: int, max_level: int
) -> Iterator[tuple[int, NDArray[np.float32]]]:
    """Create a pyramid of images by downscaling the input image by a factor of 2.

    Args:
        image (NDArray): Input image.
        level (int): Level of the input image.
        max_level (int): Maximum level of the pyramid.

    Returns:
        Iterable pyramid of images.
    """
    for level in range(level, max_level):
        yield level, image.astype(np.float32)
        image = downscale_local_mean(image, 2)
    return max_level, image.astype(np.float32)


async def put_level(
    pixelmap: NDArray[Any],
    session: aiohttp.ClientSession,
    job_id: str,
    pixelmap_id: str,
    level: int,
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
) -> None:
    tasks = []
    for tile_y in range(min_y, max_y + 1):
        for tile_x in range(min_x, max_x + 1):
            x = tile_x * 256
            y = tile_y * 256
            tile = pixelmap[y : y + 256, x : x + 256]
            tasks.append(
                session.put(
                    f"/v3/{job_id}/pixelmaps/{pixelmap_id}/level/{level}/position/{tile_x}/{tile_y}/data",
                    data=np.pad(
                        tile, ((0, 256 - tile.shape[0]), (0, 256 - tile.shape[1]))
                    ).tobytes(),
                )
            )

    await asyncio.gather(*tasks)
