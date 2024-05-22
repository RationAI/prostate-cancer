from pathlib import Path
from typing import Any

import pyvips
from numpy.typing import NDArray
from ray import serve

from app.empaia import Client
from app.empaia.typing import WSI


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
        "memory": 2000 * 1024 * 1024,  # quota 2 GiB
    },
)
class UploadService:
    """Upload service for uploading pixelmaps to EMPIA."""

    async def __call__(
        self,
        client: Client,
        wsi: WSI,
        name: str,
        key: str,
        array: NDArray[Any],
        level: int,
    ) -> None:
        pyvips.cache_set_max_mem(1500 * 1024 * 1024)  # 1.5 GiB

        path = Path(f"/mnt/data/wsi_mask/{wsi['id']}/{key}.tiff")
        path.parent.mkdir(parents=True, exist_ok=True)

        extent = wsi["levels"][level]["extent"]

        image = pyvips.Image.new_from_array(array)
        image = image.resize(
            extent["x"] / image.width,
            vscale=extent["y"] / image.height,
            kernel="nearest",
        )

        image.tiffsave(
            path,
            bigtiff=True,
            compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
            tile=True,
            tile_width=256,
            tile_height=256,
            pyramid=True,
        )

        async with client:
            response = await client.post_wsi_mask(
                wsi["id"], str(path.relative_to("/mnt/data"))
            )
            print(f"Writing mask: {response['id']}")

            data = {
                "name": name,
                "type": "string",
                "value": response["id"],
                "creator_type": "job",
                "creator_id": client.job_id,
            }
            await client.post_output(key, data)
