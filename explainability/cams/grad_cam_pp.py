import torch
from jaxtyping import Float
import numpy as np


# from explainability.cams.abstract import AbstractCAMHook


# class GradCAMPlusPlusBatchHook(AbstractCAMHook):
#     """Batch-wise Grad-CAM++ hook (approximation using first-order grads).

#     Computes alpha weights per location per channel using grad powers as in the Grad-CAM++ paper.
#     """

#     def __init__(self, *args, eps: float = 1e-6, **kwargs):
#         super().__init__(*args, **kwargs, method_name="grad_cam_pp")
#         self.eps = eps

#     def _compute_cams(self, grad):
#         activations = torch.relu(self._activations)  # ensure non-negative activations
#         g = grad
#         g2 = g * g
#         g3 = g2 * g
#         sum_a = activations.sum(dim=(2, 3), keepdim=True)  # [B,C,1,1]
#         alpha_num = g2
#         alpha_den = 2.0 * g2 + sum_a * g3 + self.eps
#         alpha = alpha_num / alpha_den  # [B,C,H,W]
#         weights = (alpha * torch.relu(g)).sum(dim=(2, 3), keepdim=True)  # [B,C,1,1]
#         cams = (weights * activations).sum(dim=1)  # [B,H,W]
#         cams = torch.clamp(cams, min=0)
#         return cams


def grad_cam_pp(
    activations: Float[torch.Tensor, "B C H W"],
    gradients: Float[torch.Tensor, "B C H W"],
    eps: float = 1e-6,
) -> Float[torch.Tensor, "B H W"]:
    """Compute Grad-CAM++ maps given activations and gradients.

    Args:
        activations: Activation maps from the target layer, shape [B, C, H, W].
        gradients: Gradients w.r.t. the activations, shape [B, C, H, W].
        eps: Small value to avoid division by zero.

    Returns:
        cams: Grad-CAM++ maps, shape [B, H, W].
    """
    activations = torch.relu(activations)
    g = gradients
    g2 = g * g
    g3 = g2 * g
    sum_a = activations.sum(dim=(2, 3), keepdim=True)  # [B,C,1,1]
    alpha_num = g2
    alpha_den = 2.0 * g2 + sum_a * g3 + eps
    alpha = alpha_num / alpha_den  # [B,C,H,W]
    weights = (alpha * torch.relu(g)).sum(dim=(2, 3), keepdim=True)  # [B,C,1,1]
    cams = (weights * activations).sum(dim=1)  # [B,H,W]
    cams = torch.clamp(cams, min=0)
    return cams

def grad_cam_pp_numpy(
    activations: np.ndarray,
    gradients: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute Grad-CAM++ maps given activations and gradients.

    Args:
        activations: Activation maps from the target layer, shape [B, C, H, W].
        gradients: Gradients w.r.t. the activations, shape [B, C, H, W].
        eps: Small value to avoid division by zero.

    Returns:
        cams: Grad-CAM++ maps, shape [B, H, W].
    """
    activations = np.maximum(activations, 0)
    # g = gradients
    g2 = gradients * gradients
    # g3 = g2 * gradients
    # sum_a = activations.sum(axis=(2, 3), keepdims=True)  # [B,C,1,1]
    # alpha_num = g2
    # alpha_den = 2.0 * g2 + sum_a * g3 + eps
    # alpha = g2 / alpha_den  # [B,C,H,W]
    # weights = ((g2 / alpha_den) * np.maximum(gradients, 0)).sum(axis=(2, 3), keepdims=True)  # [B,C,1,1]
    # cams = ((((g2 / alpha_den) * np.maximum(gradients, 0)).sum(axis=(2, 3), keepdims=True)) * activations).sum(axis=1)  # [B,H,W]
    # cams = np.clip((((((g2 / alpha_den) * np.maximum(gradients, 0)).sum(axis=(2, 3), keepdims=True)) * activations).sum(axis=1)), a_min=0, a_max=None)
    return np.clip((((((g2 / (2.0 * g2 + (activations.sum(axis=(2, 3), keepdims=True)) * (g2 * gradients) + eps)) * np.maximum(gradients, 0)).sum(axis=(2, 3), keepdims=True)) * activations).sum(axis=1)), a_min=0, a_max=None)