import numpy as np
from numpy.typing import NDArray
from ray.serve.handle import (
    DeploymentHandle,
    DeploymentResponse,
    DeploymentResponseGenerator,
)

from app.empaia import Client
from app.empaia.typing import WSI


async def put_background_mask(
    client: Client,
    tiling_service: DeploymentHandle,
    wsi: WSI,
    mask: NDArray[np.float32],
    min_level: int,
) -> DeploymentResponse | DeploymentResponseGenerator:
    pixelmap = {
        "name": "Background mask",
        "reference_id": wsi["id"],
        "reference_type": "wsi",
        "creator_id": client.job_id,
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
                "class_value": "org.empaia.rationai.prostate_cancer.v3.0.classes.background",
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
        for i, level in enumerate(wsi["levels"][min_level:], start=min_level)
    ]
    pixelmap = await client.post_output("background_mask", pixelmap)

    return tiling_service.remote(
        client.api_url, client.job_id, client.token, wsi, pixelmap, mask
    )
