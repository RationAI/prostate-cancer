import pyvips
import torch
from jaxtyping import Float
from torchvision import transforms
import numpy as np
from pathlib import Path
from rationai.masks import write_big_tiff


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


def save_image_xopat_compatible(
    image: torch.Tensor | np.ndarray,
    save_path: Path, 
    target_extent_x: int, 
    target_extent_y: int,
    microns_per_pixel_x: float, 
    microns_per_pixel_y: float,
):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    print("IMG SHP:", image.shape)
    level_size_multiplier_x = target_extent_x / image.shape[1] 
    level_size_multiplier_y = target_extent_y / image.shape[0]
    print("Multipliers:", level_size_multiplier_x, level_size_multiplier_y)
    vips_im = pyvips.Image.new_from_array(image).affine(
        (level_size_multiplier_x, 0, 0, level_size_multiplier_y), interpolate=pyvips.Interpolate.new("nearest")
    ).cast("uchar")

    # TODO: this might be wrong, but who cares, I don't
    new_mpp_x = microns_per_pixel_x / level_size_multiplier_x
    new_mpp_y = microns_per_pixel_y / level_size_multiplier_y

    write_big_tiff(
        vips_im,
        save_path,
        mpp_x=new_mpp_x,
        mpp_y=new_mpp_y,
    )