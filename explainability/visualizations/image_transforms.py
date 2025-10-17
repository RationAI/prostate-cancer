import pyvips
import torch
from jaxtyping import Float
from torchvision import transforms
import numpy as np
from pathlib import Path


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


def save_image_xopat_compatible(image: torch.Tensor | np.ndarray, save_path: Path, target_extent_x: int, target_extent_y: int):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    print("IMG SHP:", image.shape)
    level_size_multiplier_x = target_extent_x / image.shape[0] 
    level_size_multiplier_y = target_extent_y / image.shape[1]
    print("Multipliers:", level_size_multiplier_x, level_size_multiplier_y)
    vips_im = pyvips.Image.new_from_array(image).affine(
        (level_size_multiplier_x, 0, 0, level_size_multiplier_y), interpolate=pyvips.Interpolate.new("nearest")
    ).cast("uchar")
    
    # print("preparing graph")


    vips_im.tiffsave(
        save_path,
        bigtiff=True,
        compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,
    )