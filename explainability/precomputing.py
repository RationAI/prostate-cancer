from numpy.lib.format import open_memmap
from pathlib import Path
import math
import numpy as np
import torch

import matplotlib.pyplot as plt


class MultichannelHeatmapAssembler:
    def __init__(
        self,
        heatmap_width: int,
        heatmap_height: int,
        heatmap_channels: int,
        # tile_extent: int,
        # step_size: int,
        npy_file_path: Path,
    ):
        # self.original_heatmap_width = heatmap_width
        # self.original_heatmap_height = heatmap_height
        # # self.heatmap_channels = heatmap_channels

        # # check that the tile_extent with the step size can tile the heatmap fully,
        # # and if not adjust the heatmap size to be fully tiled
        # remainder_x = (heatmap_width - tile_extent) % step_size
        # remainder_y = (heatmap_height - tile_extent) % step_size
        # if remainder_x != 0:
        #     heatmap_width += step_size - remainder_x
        #     print(f"Adjusted heatmap width to {heatmap_width} to fit tiles")
        # if remainder_y != 0:
        #     heatmap_height += step_size - remainder_y
        #     print(f"Adjusted heatmap height to {heatmap_height} to fit tiles")

        

        # self.gcd_size_factor = math.gcd(tile_extent, step_size)
        # self.overlap_counter_tile_size = tile_extent // self.gcd_size_factor
        # self.accumulator_tile_size = tile_extent
        # self.level_coord_multiplier = 2 ** sample_level

        # Calculate sizes for accumulator and overlap counter
        # counter_w = heatmap_width // self.gcd_size_factor
        # counter_h = heatmap_height // self.gcd_size_factor

        # init using lib.format.open_memmap
        self.heatmap_accumulator = open_memmap(
            npy_file_path,
            mode="w+",
            dtype="float32",
            shape=(heatmap_channels, heatmap_height, heatmap_width),
        )

        # # overlap counter should fit in memory
        # self.patch_overlap_counter = np.zeros(
        #     (1, counter_h, counter_w),  # row-first format (C, H, W)
        #     dtype=np.uint8,
        # ) 
        # overlap counter should fit in memory
        self.patch_overlap_counter = np.zeros(
            (1, heatmap_height, heatmap_width),  # row-first format (C, H, W)
            dtype=np.uint8,
        )
        print("SHAPES:", self.patch_overlap_counter.shape, self.heatmap_accumulator.shape)

    def update_batch_torch(self, data: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> None:
        """Expects the data in the form (B, C, H, W)"""

        # compress overlap counter coordinates
        # xs_count = xs // self.gcd_size_factor
        # ys_count = ys // self.gcd_size_factor

        # Paste tiles onto mask
        # do not try to optimize this, not worth it. You would have to resolve issues with overlaps preventing direct broadcasting,
        # which would require a lot of extra logic and probably end up not being significantly faster anyway.
        # for xa, ya, xc, yc, tile in zip(
        #     xs, ys, xs_count, ys_count, data, strict=True
        # ):
        for xa, ya, tile in zip(
            xs, ys, data, strict=True
        ):
            tile_w, tile_h = tile.shape[1], tile.shape[2]
            mm_c, mm_h, mm_w = self.heatmap_accumulator[
                :,
                ya : ya + tile_h,
                xa : xa + tile_w,
            ].shape

            self.heatmap_accumulator[
                :,
                ya : ya + tile_h,
                xa : xa + tile_w,
            ] += tile[:mm_c, :mm_h, :mm_w]

            # self.heatmap_accumulator[
            #     :,
            #     ya : ya + tile_h,
            #     xa : xa + tile_w,
            # ] += tile
            self.patch_overlap_counter[
                :,
                ya : ya + tile_h,
                xa : xa + tile_w,
            ] += 1

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """Finalize the heatmap assembly by normalizing the accumulated heatmap by the overlap counts."""
        # Normalize heatmap by patch overlap counts but do not expand the counter
        self.heatmap_accumulator /= self.patch_overlap_counter.clip(min=1)
        self.heatmap_accumulator.flush()
        # Return the final heatmap, cropped to the original heatmap extent
        return self.heatmap_accumulator, self.patch_overlap_counter
    
# perform sanity checks to make sure the assembler works as expected
def test_heatmap_assembler():
    heatmap_width = 256 + 7
    heatmap_height = 128 + 15
    heatmap_channels = 3
    tile_extent = 32
    npy_file_path = Path("test_heatmap.npy")

    assembler = MultichannelHeatmapAssembler(
        heatmap_width,
        heatmap_height,
        heatmap_channels,
        npy_file_path=npy_file_path
    )
    # create dummy data
    B = 4
    example_tile_batch = np.ones((B, heatmap_channels, tile_extent, tile_extent), dtype=np.float32)
    increment = example_tile_batch.sum()
    assert increment == B*heatmap_channels*tile_extent*tile_extent, f"Checksum mismatch in example tile batch: Should be {B*heatmap_channels*tile_extent*tile_extent}, is {example_tile_batch.sum()}"
    xs = []
    ys = []
    # add the tiles randomly to cover the heatmap
    for i in range(7):
        x, y = (np.random.rand(B,)*(heatmap_width - tile_extent)).astype(int), (np.random.rand(B,)*(heatmap_height - tile_extent)).astype(int)
        assembler.update_batch_torch(example_tile_batch, x, y)
        plt.imshow(assembler.heatmap_accumulator[0], cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.show()
        assert assembler.heatmap_accumulator.sum() == (i+1)*increment, f"Checksum mismatch in heatmap accumulator after update {i+1}. Is {assembler.heatmap_accumulator.sum()}, but expected {(i+1)*increment}"

    assert assembler.heatmap_accumulator.sum() == 7*B*tile_extent*tile_extent*heatmap_channels, f"Checksum mismatch in heatmap accumulator after updates. Is {assembler.heatmap_accumulator.sum()}, but expected {7*B*tile_extent*tile_extent*heatmap_channels}"

    assembled_heatmap, overlap_counter = assembler.finalize()
    assert assembled_heatmap.shape == (heatmap_channels, heatmap_height, heatmap_width), f"Assembled heatmap has incorrect shape: {assembled_heatmap.shape}"
    assert overlap_counter.shape == (1, heatmap_height, heatmap_width), f"Overlap counter has incorrect shape: {overlap_counter.shape}"
    assert overlap_counter.sum() == 7*B*tile_extent*tile_extent, f"Checksum mismatch in overlap counter after finalization. Is {overlap_counter.sum()}, but expected {7*B*tile_extent*tile_extent}"
    assert assembled_heatmap.max() == 1.0, f"Assembled heatmap values should be normalized to 1.0 after finalization: {assembled_heatmap.max()}"

test_heatmap_assembler()
    