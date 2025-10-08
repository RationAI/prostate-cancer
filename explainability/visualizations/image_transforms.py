import torch
from jaxtyping import Float
from torchvision import transforms


def get_inverse_norm_transform(
    mean: Float[torch.Tensor, "C"] | None = None,
    std: Float[torch.Tensor, "C"] | None = None,
) -> torch.nn.Module:
    """Get a torchvision transform that denormalizes images.

    Args:
        mean: Normalization mean values [C] (if None, uses default values)
        std: Normalization std values [C] (if None, uses default values)

    Returns:
        A torchvision.transforms.Normalize object that denormalizes images
    """
    if mean is None:
        mean = torch.tensor([228.5544, 178.8584, 219.8793])
    if std is None:
        std = torch.tensor([27.8285, 51.4639, 26.4458])
    inv_std = 1 / std
    inv_mean = -mean * inv_std
    return transforms.Normalize(mean=inv_mean.tolist(), std=inv_std.tolist())
