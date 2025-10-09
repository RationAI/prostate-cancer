import matplotlib.pyplot as plt
import torch

from explainability.visualizations.image_transforms import get_inverse_norm_transform
from torch import Tensor
from jaxtyping import Float, Int
import matplotlib.pyplot as plt


def get_patches_from_image_batch(indices: Int[Tensor, "K"], image_batch: Float[Tensor, "B C H W"], context_size: int | tuple[int, int], embedding_batch: Float[Tensor, "B D H2 W2"]) -> Float[Tensor, "K C (H_patch) (W_patch)"]:
    """Obtain patches from a batch of images given the indices of the patches.

    This method gets a list of indices which index into a batch embedding vectors obtained for a batch of images.
    The image is divided into patches, and each patch has a corresponding embedding vector.
    Suppose we have a batch of B images, each of size C x H x W, and the embedding vectors are of size D and there is H2 x W2 of them per image.
    The indices are in the range [B*H2*W2 - 1] and index into the flattened array of embedding vectors.
    This method will us the width and height ratio between the image and the embedding to extract the corresponding patches from the image.
    Args:
        indices: A 1D tensor of shape (K,) containing the indices of the patches to extract.
        image_batch: A 4D tensor of shape (B, C, H, W) containing the batch of images.
        context_size: An integer or tuple specifying the size of the patch to extract. If an integer is provided, it is used for both height and width.
        embedding_batch: A 4D tensor of shape (B, D, H2, W2) containing the batch of embedding vectors.
    Returns:
        A 4D tensor of shape (K, C, H_patch, W_patch) containing the extracted patches.
    """
    if isinstance(context_size, int):
        context_vertical = context_size
        context_horizontal = context_size
    else:
        context_vertical, context_horizontal = context_size

    B, C, H, W = image_batch.shape
    B2, D, H2, W2 = embedding_batch.shape
    assert B == B2, "Batch size of images and embeddings must be the same"



    height_ratio = H / H2
    width_ratio = W / W2


    patches = []
    for idx in indices:
        b = idx // (H2 * W2)
        hw_idx = idx % (H2 * W2)
        h_idx = hw_idx // W2
        w_idx = hw_idx % W2

        center_h = int((h_idx + 0.5) * height_ratio)
        center_w = int((w_idx + 0.5) * width_ratio)

        h_start = max(0, center_h - int(height_ratio //2) - context_vertical)
        h_end = min(H, h_start + int(height_ratio) + context_vertical)
        w_start = max(0, center_w - int(width_ratio//2) - context_horizontal)
        w_end = min(W, w_start + int(width_ratio) + context_horizontal)

        patch = image_batch[b : b + 1, :, h_start:h_end, w_start:w_end]
        patches.append(patch)

    return torch.cat(patches, dim=0)




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

    # images = batch_to_images(patches)
    inv_norm = get_inverse_norm_transform()
    images = [
        inv_norm(img).permute(1, 2, 0).cpu().clamp(0, 255).to(torch.uint8)
        for img in patches
    ]

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

        h, w = images[0].shape[1], images[0].shape[0]
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
