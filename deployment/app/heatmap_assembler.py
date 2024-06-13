import asyncio
import logging
import math

import numpy as np
from aiohttp import ClientConnectionError, ClientResponseError
from numpy.typing import NDArray
from rationai.empaia import Client
from rationai.empaia.typing import SlideInfo
from ray.serve.handle import DeploymentHandle


MAX_RETRIES = 5

log = logging.getLogger("ray.serve")


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
    ) -> None:
        self.model = model
        self.upload_service = upload_service
        self.client = client
        self.wsi = wsi
        self.wsi_level = wsi_level
        self.wsi_tile_size = wsi_tile_size
        self.wsi_stride = wsi_stride

        self._tile_size = wsi_tile_size // math.gcd(wsi_tile_size, wsi_stride)
        self._stride = wsi_stride // math.gcd(wsi_tile_size, wsi_stride)
        self._overlaps = self._count_overlaps(attention_mask)
        self._heatmap_accumulator = np.zeros_like(self._overlaps, dtype=np.float32)
        self._counter = np.zeros_like(self._overlaps, dtype=np.uint8)

    async def __call__(self, x: int, y: int) -> None:
        for retry in range(MAX_RETRIES):
            try:
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
                break
            except (ClientConnectionError, ClientResponseError) as e:
                if retry + 1 == MAX_RETRIES:
                    raise e
                log.warning(
                    "Retrying tile %d, %d, attempt %d", x, y, retry + 1, exc_info=e
                )
                await asyncio.sleep(2**retry)

    async def finalize(self) -> np.uint:
        await self.upload_service.remote(
            client=self.client,
            wsi=self.wsi,
            name="Probability mask",
            key="probability_mask",
            array=(self._heatmap_accumulator * 255).astype(np.uint8),
            min_level=self.wsi_level,
        )
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
