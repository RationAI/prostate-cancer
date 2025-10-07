import matplotlib.pyplot as plt
import torch

from explainability.visualizations.to_image import batch_to_images


def visualize_prototypes_row(
    patches: torch.Tensor,
    figsize_scale: float = 1.0,
    color: tuple[float, float, float] | None = None,
) -> None:
    """Visualize an optional color swatch followed by patches in a single row.

    Parameters
    - patches: [p, 3, H, W] (normalized) or [p, 1, H, W]
    - figsize_scale: scale factor for figure size
    - color: optional RGB tuple in [0,1] for the swatch (prepended tile)
    """
    if not isinstance(patches, torch.Tensor):
        raise TypeError("patches must be a torch.Tensor")

    if patches.ndim != 4:
        raise ValueError("patches must be a 4D tensor [p, C, H, W]")

    channels = patches.size(1)
    if channels == 1:
        patches = patches.repeat(1, 3, 1, 1)
    elif channels not in (3, 4):
        raise ValueError("patches must have 1, 3, or 4 channels")

    images = batch_to_images(patches)

    p = len(images)
    if p == 0:
        raise ValueError("No patches to visualize")

    if color is not None:
        import numpy as np
        from PIL import Image

        if not (
            isinstance(color, tuple)
            and len(color) == 3
            and all(isinstance(c, int | float) for c in color)
        ):
            raise TypeError("color must be an RGB tuple of floats in [0,1]")
        rgb = tuple(float(max(0.0, min(1.0, c))) for c in color)

        h, w = images[0].size[1], images[0].size[0]
        swatch = np.ones((h, w, 3), dtype=np.float32) * np.array(rgb, dtype=np.float32)
        swatch = (np.clip(swatch * 255.0, 0, 255)).astype(np.uint8)
        color_img = Image.fromarray(swatch)
        images = [color_img, *images]

    total = len(images)

    h, w = images[0].size[1], images[0].size[0]
    base_size = max(1.5, min(4.0, w / 64.0)) * figsize_scale
    fig, axes = plt.subplots(1, total, figsize=(base_size * total, base_size))
    if total == 1:
        axes = [axes]

    for ax, img in zip(axes, images, strict=False):
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
