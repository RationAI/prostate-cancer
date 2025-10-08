import matplotlib.pyplot as plt
import torch
from jaxtyping import Int, UInt8


def get_overlay_from_clustering(
    indices: Int[torch.Tensor, "B H W"],
    n_indices: int | None = None,
    colormap_lut: UInt8[torch.Tensor, "N 3"] | None = None,
) -> UInt8[torch.Tensor, "B 3 H W"]:
    """Takes as input the clustering indices for each pixel in the feature map
    and returns an overlay visualization for each image in the batch.
    Each pixel is colored according to its cluster.

    Args:
        indices: [B, H, W] tensor of cluster indices (int)
        n_indices: number of clusters (optional)
        colormap_lut: [N, 3] tensor of RGB colors (optional, uint8)

    Returns:
        overlays: [B, 3, H, W] tensor of RGB overlays (uint8)
    """
    if n_indices is None:
        n_indices = int(indices.max().item()) + 1

    if colormap_lut is None:
        # Use matplotlib to get colors, then convert to torch tensor
        colors = plt.cm.get_cmap("tab10", max(10, n_indices))
        colormap_lut = torch.tensor(
            (colors(range(n_indices))[:, :3] * 255).astype("uint8"),
            dtype=torch.uint8,
            device=indices.device,
        )  # [N, 3]

    # Compute overlays in a vectorized fashion
    B, H, W = indices.shape
    overlays = (
        colormap_lut[indices.reshape(-1)].reshape(B, H, W, 3).permute(0, 3, 1, 2)
    )  # [B, 3, H, W]
    return overlays
