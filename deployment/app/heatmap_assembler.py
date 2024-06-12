import math
from pathlib import Path

import numpy as np
import ray
from aiohttp import ClientConnectionError, ClientResponseError
from numpy.typing import NDArray
from rationai.empaia import Client
from rationai.empaia.typing import SlideInfo
from ray.serve.handle import DeploymentHandle

from app.lib.functools import cached_property


@ray.remote(
    num_cpus=0.25,
    max_restarts=-1,
    memory=700 * 1024 * 1024,  # quota 800 MiB
)
class HeatmapAssembler:
    def __init__(
        self,
        model: DeploymentHandle,
        upload_service: DeploymentHandle,
        client: Client,
        wsi: SlideInfo,
        wsi_level: int,
        wsi_tile_size: int,
        wsi_stride: int,
        attention_mask: NDArray[np.bool_],
        checkpoint_dir: Path,
    ) -> None:
        self.model = model
        self.upload_service = upload_service
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

    @ray.method(
        retry_exceptions=[ClientConnectionError, ClientResponseError],
        max_task_retries=-1,
    )
    async def __call__(self, x: int, y: int) -> None:
        tile = await self.client.get_region(
            wsi_id=self.wsi.id,
            level=self.wsi_level,
            x=x * self.wsi_stride,
            y=y * self.wsi_stride,
            height=self.wsi_tile_size,
            width=self.wsi_tile_size,
        )
        prediction = await self.model.remote(tile)

        roi = self._get_roi(x, y)
        self._counter[roi] += 1
        self._heatmap_accumulator[roi] += prediction / self._overlaps[roi]

        np.save(self._heatmap_accumulator_checkpoint, self._heatmap_accumulator)
        np.save(self._counter_checkpoint, self._counter)

    async def finalize(self) -> np.uint:
        await self.upload_service.remote(
            client=self.client,
            wsi=self.wsi,
            name="Probability mask",
            key="probability_mask",
            array=(self._heatmap_accumulator * 255).astype(np.uint8),
            level=self.wsi_level,
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
