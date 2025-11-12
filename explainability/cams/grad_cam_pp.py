from pathlib import Path
import torch
from jaxtyping import Float
import numpy as np
import tempfile



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
    activations: Float[np.ndarray, "C H W"],
    gradients: Float[np.ndarray, "C H W"],
    eps: float = 1e-6,
) -> Float[np.ndarray, "H W"]:
    """Compute Grad-CAM++ maps given activations and gradients.

    Args:
        activations: Activation maps from the target layer, shape [C, H, W].
        gradients: Gradients w.r.t. the activations, shape [C, H, W].
        eps: Small value to avoid division by zero.

    Returns:
        cams: Grad-CAM++ maps, shape [H, W].
    """
    activations = np.maximum(activations, 0)
    g2 = gradients * gradients
    sum_a = activations.sum(axis=(1, 2), keepdims=True)  # [C,1,1]
    alpha_den = 2.0 * g2 + sum_a * (g2 * gradients) + eps

    return np.clip((((((g2 / alpha_den) * np.maximum(gradients, 0)).sum(axis=(1, 2), keepdims=True)) * activations).sum(axis=0)), a_min=0, a_max=None)





