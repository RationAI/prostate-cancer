import json
import math
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import ray
from aiohttp import ClientConnectionError
from numpy.typing import NDArray
from ray.serve.handle import DeploymentHandle

from app.empaia import Client
from app.empaia.typing import WSI, ContinuousPixelmap
from app.lib.functools import cached_property


@ray.remote(
    num_cpus=0.5,
    max_restarts=-1,
    memory=1000 * 1024 * 1024,  # quota 1 GiB
)
class HeatmapAssembler:
    def __init__(
        self,
        model: DeploymentHandle,
        tiling_service: DeploymentHandle,
        client: Client,
        wsi: WSI,
        wsi_level: int,
        wsi_tile_size: int,
        wsi_stride: int,
        attention_mask: NDArray[np.bool_],
        checkpoint_dir: Path,
    ) -> None:
        self.model = model
        self.tiling_service = tiling_service
        self.client = client
        self.wsi = wsi
        self.wsi_level = wsi_level
        self.wsi_tile_size = wsi_tile_size
        self.wsi_stride = wsi_stride
        self.checkpoint_dir = checkpoint_dir

        self._heatmap_accumulator_checkpoint = (
            checkpoint_dir / "heatmap_accumulator.npy"
        )
        self._counter_checkpoint = checkpoint_dir / "counter.npy"
        self._tile_size = wsi_tile_size // math.gcd(wsi_tile_size, wsi_stride)
        self._stride = wsi_stride // math.gcd(wsi_tile_size, wsi_stride)
        self._overlaps = self._count_overlaps(attention_mask)

    @cached_property
    def _heatmap_accumulator(self) -> NDArray[np.float32]:
        if self._heatmap_accumulator_checkpoint.exists():
            return np.load(self._heatmap_accumulator_checkpoint)
        return np.zeros_like(self._overlaps, dtype=np.float32)

    @cached_property
    def _counter(self) -> NDArray[np.uint8]:
        if self._counter_checkpoint.exists():
            return np.load(self._counter_checkpoint)
        return np.zeros_like(self._overlaps, dtype=np.uint8)

    @ray.method(retry_exceptions=[ClientConnectionError], max_task_retries=-1)
    async def __call__(self, x: int, y: int) -> None:
        data = await self.client.get_region(
            wsi_id=self.wsi["id"],
            level=self.wsi_level,
            x=x * self.wsi_stride,
            y=y * self.wsi_stride,
            height=self.wsi_tile_size,
            width=self.wsi_tile_size,
        )
        prediction = await self.model.remote(np.asarray(iio.imread(data)))

        roi = self._get_roi(x, y)
        self._counter[roi] += 1
        self._heatmap_accumulator[roi] += prediction / self._overlaps[roi]

        np.save(self._heatmap_accumulator_checkpoint, self._heatmap_accumulator)
        np.save(self._counter_checkpoint, self._counter)

    @ray.method(retry_exceptions=[ClientConnectionError], max_task_retries=-1)
    async def finalize(self) -> np.uint:
        pixelmap = await self._post_pixelmap()
        await self.tiling_service.remote(
            self.client.api_url,
            self.client.job_id,
            self.client.token,
            self.wsi,
            pixelmap,
            self._heatmap_accumulator,
        )
        await self.client.close()
        return np.sum(self._overlaps - self._counter)

    def _count_overlaps(self, attention_mask: NDArray[np.bool_]) -> NDArray[np.uint8]:
        height = (attention_mask.shape[0] - 1) * self._stride + self._tile_size
        width = (attention_mask.shape[1] - 1) * self._stride + self._tile_size

        overlaps = np.zeros((height, width), dtype=np.uint8)
        for (y, x), attention in np.ndenumerate(attention_mask):
            if attention:
                overlaps[self._get_roi(x, y)] += 1
        return overlaps

    def _get_roi(self, x: int, y: int) -> tuple[slice, slice]:
        return (
            slice(y * self._stride, y * self._stride + self._tile_size),
            slice(x * self._stride, x * self._stride + self._tile_size),
        )

    async def _post_pixelmap(self) -> ContinuousPixelmap:
        checkpoint = self.checkpoint_dir / "pixelmap.json"
        if checkpoint.exists():
            with open(checkpoint, "r", encoding="utf-8") as f:
                return json.load(f)

        pixelmap = {
            "name": "Probability mask",
            "reference_id": self.wsi["id"],
            "reference_type": "wsi",
            "creator_id": self.client.job_id,
            "creator_type": "job",
            "type": "continuous_pixelmap",
            "element_type": "float32",
            "min_value": 0.0,
            "neutral_value": 0.0,
            "max_value": 1.0,
            "tilesize": 1024,
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
                "slide_level": i,
                "position_min_x": 0,
                "position_min_y": 0,
                "position_max_x": (level["extent"]["x"] - 1) // pixelmap["tilesize"],
                "position_max_y": (level["extent"]["y"] - 1) // pixelmap["tilesize"],
            }
            for i, level in enumerate(
                self.wsi["levels"][self.wsi_level :], start=self.wsi_level
            )
        ]

        pixelmap = await self.client.post_output("probability_mask", pixelmap)
        with open(checkpoint, "w", encoding="utf-8") as f:
            json.dump(pixelmap, f)
        return pixelmap
