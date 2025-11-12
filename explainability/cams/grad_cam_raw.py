from pathlib import Path
import torch
from jaxtyping import Float
import numpy as np


def grad_cam_raw_numpy(
    activations: Float[torch.Tensor, "C H W"],
    gradients: Float[torch.Tensor, "C H W"],
) -> Float[torch.Tensor, "H W"]:
    """Compute raw Grad-CAM maps given activations and gradients.

    This algorithm differs from the original Grad-CAM in that it does not apply ReLU to the weighted combination of activations.
    This allows for both positive and negative contributions to contribute to the CAMs.

    Args:
        activations: Activation maps from the target layer, shape [C, H, W].
        gradients: Gradients w.r.t. the activations, shape [C, H, W].
    Returns:
        cams: Raw Grad-CAM maps, shape [H, W].
    """
    weights = gradients.mean(axis=(1, 2))  # [C]
    cams = (weights[:, np.newaxis, np.newaxis] * activations).sum(axis=0)  # [H, W]
    return cams