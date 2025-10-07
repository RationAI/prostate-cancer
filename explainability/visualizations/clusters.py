import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display
from matplotlib.patches import Rectangle

from explainability.visualizations.to_image import batch_to_images


def overlay_clustered_locations(
    x_batch: torch.Tensor,
    embeddings_locations: torch.Tensor,
    max_indices: np.ndarray,
    colors,
    activation_grid_size: tuple[int, int] | None = None,
    point_size: int = 20,
    alpha: float = 0.6,
    draw_squares: bool = True,
):
    """Draw one figure per image with three panels: (1) original image, (2) clusters-only, and (3) overlay (image + clusters). Activation cells are rendered as upscaled squares.

    Args:
        x_batch: [B, C, H, W]
        embeddings_locations: [N, 3] with (b, i, j)
        max_indices: [N] cluster id for each location
        colors: [K] RGB color for each cluster
        activation_grid_size: (h_l, w_l) of low-res activation map; inferred if None
        point_size: used only when draw_squares=False (scatter fallback)
        alpha: overlay alpha for squares on the overlay panel
        draw_squares: draw upscaled squares if True, else scatter points
    """
    assert isinstance(x_batch, torch.Tensor) and x_batch.dim() == 4, (
        "x_batch must be [B,C,H,W]"
    )
    assert (
        isinstance(embeddings_locations, torch.Tensor)
        and embeddings_locations.dim() == 2
        and embeddings_locations.size(1) == 3
    ), "embeddings_locations must be [N,3] with (b,i,j)"

    denorm_imgs = batch_to_images(x_batch.detach())
    images = [np.asarray(img) / 255.0 for img in denorm_imgs]
    num_of_images = len(images)

    locs = embeddings_locations.detach().cpu().long()
    clusters = np.asarray(max_indices).reshape(-1)
    assert locs.size(0) == clusters.shape[0], (
        "Locations and cluster assignments must align"
    )

    unique_clusters = np.unique(clusters)

    for b in range(num_of_images):
        img = images[b]
        height, width = img.shape[:2]

        # Create a new figure with three panels for this image
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
        ax_img = axs[0, 0]
        ax_clusters = axs[0, 1]
        ax_overlay = axs[0, 2]

        # Panel 1: Image only
        ax_img.imshow(img)
        ax_img.set_axis_off()
        ax_img.set_title("Image")

        # Panel 2: Clusters only (white background)
        white_bg = np.ones_like(img)
        ax_clusters.imshow(white_bg)
        ax_clusters.set_axis_off()
        ax_clusters.set_title("Clusters")

        # Panel 3: Overlay (image + clusters)
        ax_overlay.imshow(img)
        ax_overlay.set_axis_off()
        ax_overlay.set_title("Overlay")

        # Filter locations for this image
        mask_b = locs[:, 0] == b
        if mask_b.any():
            ij = locs[mask_b, 1:3]
            cl_b = clusters[mask_b.cpu().numpy()]

            # Determine activation grid size
            if activation_grid_size is not None:
                h_l, w_l = activation_grid_size
            else:
                h_l = int(ij[:, 0].max().item() + 1)
                w_l = int(ij[:, 1].max().item() + 1)

            cell_h = height / h_l
            cell_w = width / w_l

            if draw_squares:
                # Draw rectangles on clusters panel and overlay panel
                for (i_idx, j_idx), c in zip(ij.tolist(), cl_b.tolist(), strict=False):
                    top = i_idx * cell_h
                    left = j_idx * cell_w
                    # clusters-only (opaque)
                    rect1 = Rectangle(
                        (left, top),
                        width=cell_w,
                        height=cell_h,
                        linewidth=0.0,
                        edgecolor=None,
                        facecolor=colors(int(c)),
                        alpha=1.0,
                    )
                    ax_clusters.add_patch(rect1)
                    # overlay (semi-transparent)
                    rect2 = Rectangle(
                        (left, top),
                        width=cell_w,
                        height=cell_h,
                        linewidth=0.0,
                        edgecolor=None,
                        facecolor=colors(int(c)),
                        alpha=alpha,
                    )
                    ax_overlay.add_patch(rect2)
            else:
                # Scatter fallback at cell centers
                ys = (ij[:, 0].float() + 0.5) * cell_h
                xs = (ij[:, 1].float() + 0.5) * cell_w
                ys_np = ys.cpu().numpy()
                xs_np = xs.cpu().numpy()
                for c in unique_clusters:
                    mask_c = cl_b == c
                    if np.any(mask_c):
                        ax_clusters.scatter(
                            xs_np[mask_c],
                            ys_np[mask_c],
                            s=point_size,
                            c=[colors(int(c))],
                            alpha=1.0,
                        )
                        ax_overlay.scatter(
                            xs_np[mask_c],
                            ys_np[mask_c],
                            s=point_size,
                            c=[colors(int(c))],
                            alpha=alpha,
                        )

        plt.tight_layout()
        display(fig)
        plt.close(fig)
