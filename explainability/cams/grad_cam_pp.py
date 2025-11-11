from pathlib import Path
import torch
from jaxtyping import Float
import numpy as np
import tempfile


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
    activations: Float[np.ndarray, "B C H W"],
    gradients: Float[np.ndarray, "B C H W"],
    eps: float = 1e-6,
) -> Float[np.ndarray, "B H W"]:
    """Compute Grad-CAM++ maps given activations and gradients.

    Args:
        activations: Activation maps from the target layer, shape [B, C, H, W].
        gradients: Gradients w.r.t. the activations, shape [B, C, H, W].
        eps: Small value to avoid division by zero.

    Returns:
        cams: Grad-CAM++ maps, shape [B, H, W].
    """
    print(f"DEBUG: Inside grad_cam_pp_numpy with activations shape {activations.shape} and gradients shape {gradients.shape}")
    activations = np.maximum(activations, 0)
    print(f"DEBUG: After ReLU, activations min {activations.min()}, max {activations.max()}")
    # g = gradients
    g2 = gradients * gradients
    print(f"DEBUG: g2 min {g2.min()}, max {g2.max()}", flush=True)
    # g3 = g2 * gradients
    # sum_a = activations.sum(axis=(2, 3), keepdims=True)  # [B,C,1,1]
    # alpha_num = g2
    # alpha_den = 2.0 * g2 + sum_a * g3 + eps
    # alpha = g2 / alpha_den  # [B,C,H,W]
    # weights = ((g2 / alpha_den) * np.maximum(gradients, 0)).sum(axis=(2, 3), keepdims=True)  # [B,C,1,1]
    # cams = ((((g2 / alpha_den) * np.maximum(gradients, 0)).sum(axis=(2, 3), keepdims=True)) * activations).sum(axis=1)  # [B,H,W]
    # cams = np.clip((((((g2 / alpha_den) * np.maximum(gradients, 0)).sum(axis=(2, 3), keepdims=True)) * activations).sum(axis=1)), a_min=0, a_max=None)
    return np.clip((((((g2 / (2.0 * g2 + (activations.sum(axis=(2, 3), keepdims=True)) * (g2 * gradients) + eps)) * np.maximum(gradients, 0)).sum(axis=(2, 3), keepdims=True)) * activations).sum(axis=1)), a_min=0, a_max=None)


def grad_cam_pp_numpy_memmapped(
    activations: Float[np.memmap, "C H W"],
    gradients: Float[np.memmap, "C H W"],
    eps: float = 1e-6,
    out: Float[np.memmap, "H W"] | None = None,
) -> None:
    """Compute Grad-CAM++ maps given activations and gradients.

    This method uses memmap arrays to handle large data that may not fit into memory.
    The intermediate calculations are stored in temporary memmap files.

    Args:
        activations: Activation maps from the target layer, shape [C, H, W].
        gradients: Gradients w.r.t. the activations, shape [C, H, W].
        eps: Small value to avoid division by zero.
        out: memmap for the Grad-CAM++ maps, shape [H, W].
    """
    if out is None:
        raise ValueError("Output memmap 'out' must be provided.")
    with tempfile.TemporaryDirectory() as temp_dir:
        _dir_path = Path(temp_dir)

        relu_activations = np.lib.format.open_memmap(
            filename=_dir_path / "relu_activations.npy",
            mode="w+",
            dtype=activations.dtype,
            shape=activations.shape,  # (C, H, W)
        )
        np.maximum(activations, 0, out=relu_activations)
        print(f"DEBUG: relu_activations min", flush=True)
        g2 = np.lib.format.open_memmap(
            filename=_dir_path / "g2.npy",
            mode="w+",
            dtype=gradients.dtype,
            shape=gradients.shape,  # (C, H, W)
        )
        np.multiply(gradients, gradients, out=g2)
        print(f"DEBUG: g2 min", flush=True)
        sum_a = np.lib.format.open_memmap(
            filename=_dir_path / "sum_a.npy",
            mode="w+",
            dtype=relu_activations.dtype,
            shape=(relu_activations.shape[0], 1, 1),  # (C, 1, 1)
        )
        np.sum(relu_activations, axis=(1, 2), keepdims=True, out=sum_a)
        print(f"DEBUG: sum_a min", flush=True)

        # alpha_den = 2*g2 + sum_a * (g2 * gradients) + eps
        # 2 * g2 + sum_a *g2 * gradients = g2 * (2 + sum_a * gradients)

        alpha_den = np.lib.format.open_memmap(
            filename=_dir_path / "alpha_den.npy",
            mode="w+",
            dtype=gradients.dtype,
            shape=gradients.shape,  # (C, H, W)
        )
        np.multiply(sum_a, gradients, out=alpha_den)  # reuse alpha_den memmap for sum_a * gradients
        print(f"DEBUG: alpha_den min after sum", flush=True)
        
        alpha_den += 2  # add 2 to each element in sum_a
        print(f"DEBUG: sum_a min after adding 2", flush=True)
        alpha_den *= g2  # multiply by g2 to complete alpha_den calculation
        print(f"DEBUG: sum_a min after multiplying by g2", flush=True)
        alpha_den += eps # add eps to avoid div by zero
        print(f"DEBUG: alpha_den min after adding eps", flush=True)
        # weights = (g2 / alpha_den) * max(gradients, 0)
        # = g2 * clip(gradients, 0) / alpha_den
        weights = np.lib.format.open_memmap(
            filename=_dir_path / "weights.npy",
            mode="w+",
            dtype=gradients.dtype,
            shape=gradients.shape,  # (C, 1, 1)
        )
    
        np.clip(gradients, a_min=0, a_max=None, out=weights)  # reuse weights memmap for max(gradients, 0)
        weights *= g2  # weights now holds g2 * max(gradients, 0)
        weights /= alpha_den  # weights now holds (g2 * max(gradients, 0)) / alpha_den
        print(f"DEBUG: weights min after sum", flush=True)
        weights *= relu_activations
        # Combine channel weights with activations and sum over channels -> (H, W)
        np.sum(
            weights,
            axis=0,
            out=out,
        )
        print(f"DEBUG: out min after sum", flush=True)
        np.clip(out, a_min=0, a_max=None, out=out)
        print(f"DEBUG: out min after clip", flush=True)


