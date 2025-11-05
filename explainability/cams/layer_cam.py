import torch
from jaxtyping import Float
import numpy as np


# class LayerCAMBatchHook(AbstractCAMHook):
#     """Batch-wise Layer-CAM hook.

#     Computes alpha weights per location per channel using the gradient of the output with respect to the layer's input.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs, method_name="layer_cam")

#     def _compute_cams(self, grad):
#         cams = torch.relu(grad) * torch.relu(self._activations)
#         cams = cams.sum(dim=1)
#         cams = torch.relu(cams)
#         return cams


def layer_cam(
    activations: Float[torch.Tensor, "B C H W"],
    gradients: Float[torch.Tensor, "B C H W"],
) -> Float[torch.Tensor, "B H W"]:
    """Compute Layer-CAM maps given activations and gradients.

    Args:
        activations: Activation maps from the target layer, shape [B, C, H, W].
        gradients: Gradients w.r.t. the activations, shape [B, C, H, W].

    Returns:
        cams: Layer-CAM maps, shape [B, H, W].
    """
    cams = torch.relu(gradients) * torch.relu(activations)
    cams = cams.sum(dim=1)
    cams = torch.relu(cams)
    return cams


def layer_cam_numpy(
    activations: np.ndarray,
    gradients: np.ndarray,
) -> np.ndarray:
    """Compute Layer-CAM maps given activations and gradients (numpy version).

    Args:
        activations: Activation maps from the target layer, shape [B, C, H, W].
        gradients: Gradients w.r.t. the activations, shape [B, C, H, W].

    Returns:
        cams: Layer-CAM maps, shape [B, H, W].
    """
    # cams = np.maximum(gradients, 0) * np.maximum(activations, 0)
    # cams = cams.sum(axis=1)
    # cams = np.maximum(cams, 0)
    # return cams
    return np.maximum((np.maximum(gradients, 0) * np.maximum(activations, 0)).sum(axis=1), 0)