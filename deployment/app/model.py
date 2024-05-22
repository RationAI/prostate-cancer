import numpy as np
import torch
from albumentations import Normalize
from numpy.typing import NDArray
from ray import serve
from torch import Tensor


@serve.deployment(
    num_replicas="auto",
    autoscaling_config={
        "min_replicas": 0,
        "max_replicas": 8,
        "target_ongoing_requests": 10,
        "downscale_delay_s": 10,
        "upscale_delay_s": 10,
    },
    ray_actor_options={
        "num_cpus": 0.5,
        "memory": 1000 * 1024 * 1024,  # quota 1 GiB
    },
)
class Model:
    """JIT model for prostate cancer detection."""

    def __init__(self) -> None:
        self._normalize = Normalize(mean=0.5, std=0.5)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = (
            torch.jit.load("model.pt", map_location=self._device)
            .to(self._device)
            .eval()
        )

    def __call__(self, image: NDArray[np.uint8]) -> float:
        with torch.no_grad():
            outputs = self._model(self._preprocess(image))
            return outputs.squeeze(1).cpu().numpy().item()

    def _preprocess(self, image: NDArray[np.uint8]) -> Tensor:
        normalized = self._normalize(image=image)["image"]
        return (
            torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(self._device)
        )
