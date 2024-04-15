import asyncio
import math
import tempfile
from typing import Annotated, Any

import aiohttp
import numpy as np
import ray
from fastapi import Body, FastAPI
from numpy.typing import NDArray
from ray import ObjectRef, serve
from ray.serve.handle import DeploymentHandle

from prostate_cancer.background_mask import background_mask
from prostate_cancer.collector import Collector
from prostate_cancer.progress import Progress
from prostate_cancer.utils import put_level

app = FastAPI()


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.4})
@serve.ingress(app)
class Ingress:
    """Ingress deployment for the Prostate Cancer pipeline.

    Attributes:
        max_concurrent: The maximum number of concurrent tile requests.
        background_mask_level: The level at which to compute the background mask.
        model: The model deployment handle.
    """

    max_concurrent = 100
    background_mask_level = 6

    def __init__(self, model: DeploymentHandle) -> None:
        self.model = model

    def reconfigure(self, config: dict[str, Any]) -> None:
        self.max_concurrent = config.get("max_concurrent", self.max_concurrent)
        self.background_mask_level = config.get(
            "background_mask_level", self.background_mask_level
        )

    @app.post("/")
    async def root(
        self,
        api_url: Annotated[str, Body()],
        job_id: Annotated[str, Body()],
        token: Annotated[str, Body()],
    ) -> str:
        headers = {"Authorization": f"Bearer {token}"}
        async with aiohttp.ClientSession(
            api_url, headers=headers, raise_for_status=True
        ) as session:
            response = await session.get(f"/v3/{job_id}/inputs/my_wsi")
            wsi_metadata = await response.json()

            masks = await background_mask(
                session, job_id, wsi_metadata, self.background_mask_level
            )
            collector = Collector.remote(
                self.model,
                api_url,
                job_id,
                headers,
                wsi_metadata,
                masks[8],
                tempfile.mkdtemp(),
            )

            await self._put_background_mask(session, job_id, masks, wsi_metadata)
            self._process_tiles(api_url, job_id, headers, collector, masks[8])

            num_lost = await collector.finalize.remote()
            if num_lost > 0:
                await session.put(
                    f"/v3/{job_id}/failure",
                    json={"user_message": f"{num_lost} tiles lost"},
                )
                return f"{num_lost} tiles lost"

            await session.put(f"/v3/{job_id}/finalize")
            return "Success!"

    def _process_tiles(
        self,
        api_url: str,
        job_id: str,
        headers: dict[str, str],
        collector: ObjectRef,
        mask: NDArray[np.float32],
    ):
        tiles: list[tuple[int, int]] = []
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y : y + 2, x : x + 2].mean() >= 0.5:
                    tiles.append((x, y))

        progress = Progress.options(max_concurrency=1).remote(
            api_url, job_id, headers, len(tiles), tempfile.mkdtemp()
        )
        pending: list[ObjectRef[None]] = []
        for x, y in tiles:
            if len(pending) > self.max_concurrent:
                done, pending = ray.wait(pending, num_returns=1)
                progress.update.remote(len(done))
            pending.append(collector.__call__.remote(x, y))
        ray.get(pending)
        ray.get(progress.finalize.remote())

    async def _put_background_mask(
        self,
        session: aiohttp.ClientSession,
        job_id: str,
        masks: dict[int, NDArray[np.float32]],
        wsi_metadata: dict[str, Any],
    ) -> None:
        pixelmap = {
            "name": "Background mask",
            "reference_id": wsi_metadata["id"],
            "reference_type": "wsi",
            "creator_id": job_id,
            "creator_type": "job",
            "type": "continuous_pixelmap",
            "element_type": "float32",
            "min_value": 0.0,
            "neutral_value": 0.0,
            "max_value": 1.0,
            "tilesize": 256,
            "channel_count": 1,
            "channel_class_mapping": [
                {
                    "number_value": 0,
                    "class_value": "org.empaia.rationai.prostate_cancer.v3.0.classes.background",
                }
            ],
            "levels": [
                {
                    "slide_level": level,
                    "position_min_x": 0,
                    "position_min_y": 0,
                    "position_max_x": math.ceil(mask.shape[1] / 256) - 1,
                    "position_max_y": math.ceil(mask.shape[0] / 256) - 1,
                }
                for level, mask in masks.items()
            ],
        }

        response = await session.post(
            f"/v3/{job_id}/outputs/background_mask", json=dict(pixelmap)
        )
        metadata = await response.json()

        tasks = []
        for i, (level, mask) in enumerate(masks.items()):
            if level >= 8:  # due to current xOpath limitation
                tasks.append(
                    put_level(
                        mask,
                        session,
                        job_id,
                        metadata["id"],
                        level,
                        pixelmap["levels"][i]["position_min_x"],
                        pixelmap["levels"][i]["position_min_y"],
                        pixelmap["levels"][i]["position_max_x"],
                        pixelmap["levels"][i]["position_max_y"],
                    )
                )
        await asyncio.gather(*tasks)
