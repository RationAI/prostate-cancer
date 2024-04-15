import asyncio
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import ray
from numpy.typing import NDArray
from PIL import Image
from prostate_cancer.utils import bbox, create_pyramid, put_level
from ray.serve.handle import DeploymentHandle


@ray.remote(num_cpus=0.5, max_restarts=-1)
class Collector:
    level = 8

    def __init__(
        self,
        model: DeploymentHandle,
        api_url: str,
        job_id: str,
        headers: dict[str, str],
        wsi_metadata: dict[str, Any],
        background_mask: NDArray[np.float32],
        checkpoint_dir: str,
    ) -> None:
        self.model = model
        self.job_id = job_id
        self.wsi_metadata = wsi_metadata
        self.checkpoint_dir = checkpoint_dir

        self._session = aiohttp.ClientSession(
            api_url, headers=headers, raise_for_status=True
        )

        self._probability_mask_checkpoint = Path(checkpoint_dir) / "mask.npy"
        self._counter_checkpoint = Path(checkpoint_dir) / "counter.npy"
        self._overlaps_checkpoint = Path(checkpoint_dir) / "overlaps.npy"

        self._probability_mask = (
            np.load(self._probability_mask_checkpoint)
            if self._probability_mask_checkpoint.exists()
            else np.zeros(background_mask.shape, dtype=np.float32)
        )
        self._overlaps = (
            np.load(self._overlaps_checkpoint)
            if self._overlaps_checkpoint.exists()
            else count_overlaps(background_mask)
        )
        self._counter = (
            np.load(self._counter_checkpoint)
            if self._counter_checkpoint.exists()
            else np.zeros(background_mask.shape, dtype=np.uint8)
        )

    @ray.method(retry_exceptions=["aiohttp.ClientError"], max_task_retries=5)
    async def __call__(self, x: int, y: int) -> None:
        step = self.level**2
        response = await self._session.get(
            f'/v3/{self.job_id}/regions/{self.wsi_metadata["id"]}/level/0/start/{x * step}/{y * step}/size/512/512'
        )
        image = np.array(Image.open(BytesIO(await response.read())))
        probability = await self.model.remote(image)

        tile_slice = slice(y, y + 2), slice(x, x + 2)
        self._counter[tile_slice] += 1
        self._probability_mask[tile_slice] += probability / self._overlaps[tile_slice]

        np.save(self._probability_mask_checkpoint, self._probability_mask)
        np.save(self._counter_checkpoint, self._counter)
        np.save(self._overlaps_checkpoint, self._overlaps)

    async def finalize(self) -> np.uint:
        pixelmap = {
            "name": "Probability mask",
            "reference_id": self.wsi_metadata["id"],
            "reference_type": "wsi",
            "creator_id": self.job_id,
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
                    "class_value": "org.empaia.rationai.prostate_cancer.v3.0.classes.cancer_probability",
                }
            ],
        }
        pixelmap["levels"] = [
            {
                "slide_level": level,
                "position_min_x": (box := bbox(self._overlaps))[0]
                // pixelmap["tilesize"],
                "position_min_y": box[1] // pixelmap["tilesize"],
                "position_max_x": box[2] // pixelmap["tilesize"],
                "position_max_y": box[3] // pixelmap["tilesize"],
            }
            for level in range(self.level, len(self.wsi_metadata["levels"]))
        ]

        response = await self._session.post(
            f"/v3/{self.job_id}/outputs/probability_mask", json=dict(pixelmap)
        )
        metadata = await response.json()

        tasks = []
        for i, (level, mask) in enumerate(
            create_pyramid(
                self._probability_mask, self.level, len(self.wsi_metadata["levels"])
            )
        ):
            tasks.append(
                put_level(
                    mask,
                    self._session,
                    self.job_id,
                    metadata["id"],
                    level,
                    pixelmap["levels"][i]["position_min_x"],
                    pixelmap["levels"][i]["position_min_y"],
                    pixelmap["levels"][i]["position_max_x"],
                    pixelmap["levels"][i]["position_max_y"],
                )
            )

        await asyncio.gather(*tasks)
        await self._session.close()
        shutil.rmtree(self.checkpoint_dir)
        return np.sum(self._overlaps - self._counter)


def count_overlaps(mask: NDArray[np.float32]) -> NDArray[np.uint8]:
    overlaps = np.zeros(mask.shape, dtype=np.uint8)
    for row in range(mask.shape[0] - 1):
        for col in range(mask.shape[1] - 1):
            tile_slice = slice(row, row + 2), slice(col, col + 2)
            if mask[tile_slice].mean() >= 0.5:
                overlaps[tile_slice] += 1
    return overlaps
