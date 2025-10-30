import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
import numpy as np


class ColorPalette:
    """Utility class for color palettes used in visualizations.

    This class provides tools to get color palettes in different formats, such as hex strings, RGB tuples and LUT arrays.
    The colormaps are stored as LUT (Look-Up Table) arrays by default for easy integration with image overlays.
    
    Args:
        palette: Either the name of a matplotlib colormap or a list of hex color strings.
    """

    def __init__(self, palette: str | list[str] | None = "tab10") -> None:
        if isinstance(palette, str):
            cmap = plt.get_cmap(palette)
            self.colors_lut = (
                (cmap(range(cmap.N))[:, :3] * 255).astype("uint8")
            )  # [N, 3] uint8
        elif isinstance(palette, list):
            self.colors_lut = np.array(
                [to_rgb(hex_color) for hex_color in palette],
            )  # [N, 3] uint8
        else:
            raise ValueError("palette must be a string or list of hex color strings.")
        
    def get_hex_colors(self) -> list[str]:
        """Get the color palette as a list of hex color strings.

        Returns:
            List of hex color strings.
        """
        return [to_hex(color / 255) for color in self.colors_lut]
    def get_rgb_lut(self) -> np.ndarray:
        """Get the color palette as an RGB LUT array.

        Returns:
            [N, 3] array of RGB colors (uint8).
        """
        return self.colors_lut
            
COLOR_PALETTE_ADAM = ["#000000", "#6B00F5", "#324B47", "#3D324B", "#754B3B", "#A39753", "#2EA32E", "#2E55A3", "#B82E75", "#01F5CC", "#F54701", "#F5F201"]