import asyncio
from typing import Any

import pyvips
from numpy.typing import NDArray
from ray import serve

from app.empaia import Client
from app.empaia.typing import WSI, ContinuousPixelmap, PixelmapLevel


@serve.deployment(
    num_replicas="auto",
    autoscaling_config={
        "min_replicas": 0,
        "max_replicas": 1,
        "target_ongoing_requests": 1,
        "downscale_delay_s": 10,
        "upscale_delay_s": 10,
    },
    ray_actor_options={
        "num_cpus": 1.5,
        "memory": 1000 * 1024 * 1024,  # quota 1 GiB
    },
)
class TilingService:
    """Tiling service for uploading tiles to EMPIA."""

    async def __call__(
        self,
        api_url: str,
        job_id: str,
        token: str,
        wsi: WSI,
        pixelmap: ContinuousPixelmap,
        array: NDArray[Any],
    ) -> None:
        pyvips.cache_set_max_mem(500 * 1024 * 1024)  # 500 MiB
        image = pyvips.Image.new_from_array(array)

        async with Client(api_url, job_id, token) as client:
            tasks = []
            for level in pixelmap["levels"]:
                tasks.append(self._put_level(client, wsi, pixelmap, image, level))
            await asyncio.gather(*tasks)

    async def _put_level(
        self,
        client: Client,
        wsi: WSI,
        pixelmap: ContinuousPixelmap,
        image: pyvips.Image,
        level: PixelmapLevel,
    ) -> None:
        tilesize = pixelmap["tilesize"]
        pixelmap_id = pixelmap["id"]
        extent = wsi["levels"][level["slide_level"]]["extent"]

        # Resize to match the slide level
        image = image.resize(
            extent["x"] / image.width,
            vscale=extent["y"] / image.height,
            kernel="nearest",
        )
        # Add padding
        image = image.embed(
            0,
            0,
            (level["position_max_x"] - level["position_min_x"] + 1) * tilesize,
            (level["position_max_y"] - level["position_min_y"] + 1) * tilesize,
            background=0,
        )

        for tile_y in range(level["position_min_y"], level["position_max_y"] + 1):
            for tile_x in range(level["position_min_x"], level["position_max_x"] + 1):
                await client.put_tile(
                    pixelmap_id=pixelmap_id,
                    level=level["slide_level"],
                    tile_x=tile_x,
                    tile_y=tile_y,
                    data=image.crop(
                        tile_x * tilesize, tile_y * tilesize, tilesize, tilesize
                    ).tiffsave_buffer(
                        compression=pyvips.enums.ForeignTiffCompression.DEFLATE
                    ),
                )
