import torch

from explainability.typing import EmbeddingsLocs


def find_top_prototypes(
    indices: torch.Tensor,
    images: torch.Tensor,
    locations: EmbeddingsLocs,
    ratio: int,
    width: int = 1,
    height: int = 1,
) -> torch.Tensor:
    """Extract ratio xratio RGB patches around (i, j) for selected indices.

    Parameters
    - indices: [p] indices into locations (and embeddings) selecting top representatives
    - images: [B, 3, H, W]
    - locations: [K, 3] with (b, i, j) for each embedding/location
    - ratio: side length of the square patch to extract. Also used as the grid cell size, i.e.,
      the pixel slice is [i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio].
    - width: width of the patch to extract.
    - height: height of the patch to extract.

    Returns:
    - patches: [p, 3, ratio, ratio]
    """
    if not isinstance(indices, torch.Tensor):
        raise TypeError("indices must be a torch.Tensor")
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")
    if not isinstance(locations, torch.Tensor):
        raise TypeError("locations must be a torch.Tensor")
    if locations.ndim != 2 or locations.size(-1) != 3:
        raise ValueError("locations must be [K,3] with (b,i,j)")
    if ratio <= 0:
        raise ValueError("ratio must be a positive integer (side length)")

    # Expect images as [B, 3, H, W]
    if images.dim() != 4 or images.size(1) != 3:
        raise ValueError("images must be a 4D RGB tensor [B,3,H,W]")

    batch_size, channels, image_height, image_width = (
        images.size(0),
        images.size(1),
        images.size(2),
        images.size(3),
    )

    sel = locations[indices]  # [p, 3]
    b_idx = sel[:, 0].long()
    i_idx = sel[:, 1].long()
    j_idx = sel[:, 2].long()

    if not ((b_idx >= 0).all() and (b_idx < batch_size).all()):
        raise ValueError("Batch indices in locations out of bounds for provided images")

    side = int(ratio)
    left = width // 2
    right = width - left
    top = height // 2
    bottom = height - top
    patches = images.new_zeros((indices.numel(), channels, side * width, side * height))

    for k_idx in range(indices.numel()):
        b = b_idx[k_idx].item()
        ci = i_idx[k_idx].item()
        cj = j_idx[k_idx].item()

        # Grid-aligned slice bounds
        src_top = max((ci - top) * side, 0)
        src_left = max((cj - left) * side, 0)
        src_bottom = min((ci + bottom) * side, image_height)
        src_right = min((cj + right) * side, image_width)

        crop = images[b, :, src_top:src_bottom, src_left:src_right]

        # Place crop into fixed-size output (pad if near borders)
        h_slice = src_bottom - src_top
        w_slice = src_right - src_left
        patches[k_idx, :, 0:h_slice, 0:w_slice] = crop

    return patches
