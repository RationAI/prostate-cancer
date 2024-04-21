import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any, Final

import numpy as np
import ray
from fastapi import Body, FastAPI, Response, status
from numpy.typing import NDArray
from ray import ObjectRef, serve
from ray.serve.handle import DeploymentHandle
from skimage.util import view_as_windows

from app.background_mask import get_background_mask, put_background_mask
from app.empaia import Client
from app.empaia.typing import WSI
from app.heatmap_assembler import HeatmapAssembler
from app.utils import Progress, find_closes_resolution_level

app = FastAPI()


@serve.deployment(
    num_replicas=1,
    ray_actor_options={
        "num_cpus": 0.4,
        "memory": 1000 * 1024 * 1024,  # quota 1 GiB
    },
)
@serve.ingress(app)
class Ingress:
    """Ingress deployment for the Prostate Cancer pipeline.

    Attributes:
        max_concurrent: The maximum number of concurrent tile requests.
        background_mask_level: The level at which to compute the background mask. If
            not presented, the highest level is used.
        tissue_percentage_threshold: The minimum percentage of tissue required for a
            tile to be considered.
        target_resolution: The target resolution for the model input tiles (μm per pixel).
        tile_size: The size of the model input tiles (pixels).
        stride: The stride of the model input tiles (pixels). Overlapping tiles are averaged.
    """

    max_concurrent_tiles = 100
    background_mask_level = 6
    tissue_percentage_threshold = 0.5
    target_resolution = 0.45
    tile_size = 512
    stride = 256

    def __init__(
        self, model: DeploymentHandle, tiling_service: DeploymentHandle
    ) -> None:
        self.model = model
        self.tiling_service = tiling_service

    def reconfigure(self, config: dict[str, Any]) -> None:
        self.max_concurrent_tiles = config.get(
            "max_concurrent_tiles", self.max_concurrent_tiles
        )
        self.background_mask_level = config.get(
            "background_mask_level", self.background_mask_level
        )
        self.tissue_percentage_threshold = config.get(
            "tissue_percentage_threshold", self.tissue_percentage_threshold
        )
        self.target_resolution = config.get("target_resolution", self.target_resolution)

    @app.put("/", status_code=status.HTTP_201_CREATED, response_class=Response)
    async def root(
        self,
        api_url: Annotated[str, Body()],
        job_id: Annotated[str, Body()],
        token: Annotated[str, Body()],
    ) -> None:
        checkpoint_dir = Path(tempfile.mkdtemp())

        async with Client(api_url, job_id, token) as client:
            progress = Progress.options(max_concurrency=1).remote(
                client, checkpoint_dir
            )

            wsi: Final[WSI] = await client.get_input("my_wsi")
            wsi_level: Final[int] = find_closes_resolution_level(
                levels=wsi["levels"],
                pixel_size_nm=wsi["pixel_size_nm"],
                target_resolution=self.target_resolution,
            )

            background_mask, background_mask_level = await get_background_mask(
                client, wsi, self.background_mask_level
            )

            background_mask_response = await put_background_mask(
                client=client,
                tiling_service=self.tiling_service,
                wsi=wsi,
                mask=background_mask,
                min_level=wsi_level,
            )

            attention_mask = self._get_attention_mask(
                background_mask,
                wsi["levels"][background_mask_level]["downsample_factor"]
                / wsi["levels"][wsi_level]["downsample_factor"],
            )

            heatmap_assembler = HeatmapAssembler.remote(
                self.model,
                self.tiling_service,
                client,
                wsi,
                wsi_level,
                self.tile_size,
                self.stride,
                attention_mask,
                checkpoint_dir,
            )
            progress.add.remote(0.05)

            self._process_tissue(progress, heatmap_assembler, attention_mask)

            num_lost = await heatmap_assembler.finalize.remote()
            await background_mask_response
            await progress.finalize.remote()

            if num_lost == 0:
                await client.put_finalize()
            else:
                await client.put_failure(f"{num_lost} tiles lost")

        shutil.rmtree(checkpoint_dir)

    def _get_attention_mask(
        self, mask: NDArray[np.float32], downsample_factor: float
    ) -> NDArray[np.bool_]:
        size = int(self.tile_size // downsample_factor)
        kernel = np.full((size, size), 1 / size**2, dtype=np.float32)
        stride = int(self.stride // downsample_factor)

        # 2D convolution with a stride
        arr4d = view_as_windows(mask, kernel.shape, step=stride)
        attention = np.tensordot(arr4d, kernel, axes=((2, 3), (0, 1)))
        return attention >= self.tissue_percentage_threshold

    def _process_tissue(
        self,
        progress: ObjectRef,
        heatmap_assembler: ObjectRef,
        attention_mask: NDArray[np.bool_],
    ) -> None:
        progress_weight: Final[float] = 0.95

        indices = np.nonzero(attention_mask)
        total_tiles = len(indices[0])

        pending: list[ObjectRef[None]] = []
        for y, x in zip(*indices):
            if len(pending) > self.max_concurrent_tiles:
                done, pending = ray.wait(pending, num_returns=1)
                progress.add.remote(len(done) / total_tiles * progress_weight)
            pending.append(heatmap_assembler.__call__.remote(x, y))
        ray.get(pending)
        progress.add.remote(len(pending) / total_tiles * progress_weight)
