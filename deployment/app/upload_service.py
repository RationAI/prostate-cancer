import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyvips
from numpy.typing import NDArray
from rationai.empaia import Client
from rationai.empaia.typing import DataCreatorType, SlideInfo, primitives
from ray import serve

log = logging.getLogger(__name__)


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
class UploadService:
    """Upload service for uploading slides to EMPIA."""

    output_dir = Path("/mnt/data/wsi_mask")

    def reconfigure(self, config: dict[str, Any]) -> None:
        self.output_dir = Path(config.get("output_dir", self.output_dir))

    async def __call__(
        self,
        client: Client,
        wsi: SlideInfo,
        name: str,
        key: str,
        array: NDArray[Any],
        level: int,
    ) -> None:
        pyvips.cache_set_max_mem(1000 * 1024 * 1024)  # 1 GiB

        case_id = await client.get_case_id()
        path = self.output_dir / str(case_id) / wsi.id / key / f"{uuid4()}.tiff"
        path.parent.mkdir(parents=True, exist_ok=True)

        image = pyvips.Image.new_from_array(array)
        image = image.resize(
            wsi.levels[level].extent.x / image.width,
            vscale=wsi.levels[level].extent.y / image.height,
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
                wsi.id, path.relative_to("/mnt/data").as_posix()
            )
            log.info("Writing mask: %s", response.id)

            data = primitives.PostStringPrimitive(
                name=name,
                type="string",
                value=response.id,
                creator_type=DataCreatorType.JOB,
                creator_id=client.job_id,
            )
            await client.post_output(key, data)
