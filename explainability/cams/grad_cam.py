import torch
from jaxtyping import Float

from explainability.cams.abstract import AbstractCAMHook


class GradCAMBatchHook(AbstractCAMHook):
    """Batch-wise Grad-CAM hook (approximation using first-order grads).

    Computes alpha weights per location per channel using grad powers as in the Grad-CAM paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, method_name="grad_cam")

    def _compute_cams(self, grad):
        weights = grad.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cams = (weights * self._activations).sum(dim=1)  # [B,H,W]
        cams = torch.relu(cams)
        return cams


def grad_cam(
    activations: Float[torch.Tensor, "B C H W"],
    gradients: Float[torch.Tensor, "B C H W"],
) -> Float[torch.Tensor, "B H W"]:
    """Compute Grad-CAM maps given activations and gradients.

    Args:
        activations: Activation maps from the target layer, shape [B, C, H, W].
        gradients: Gradients w.r.t. the activations, shape [B, C, H, W].

    Returns:
        cams: Grad-CAM maps, shape [B, H, W].
    """
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
    cams = (weights * activations).sum(dim=1)  # [B,H,W]
    cams = torch.relu(cams)
    return cams
