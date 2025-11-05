import numpy as np
from pathlib import Path
from explainability.precomputing import EdgeClippingMultiscaleHeatmapAssembler


def test_edge_clipping_heatmap_assembler(tmp_path):
    """Pytest for EdgeClippingMultiscaleHeatmapAssembler.

    Uses a temporary .npy file via pytest tmp_path and avoids displaying plots.
    The sizes are reduced slightly to keep the test fast while exercising the
    clipping and accumulation logic.
    """
    np.random.seed(0)

    heatmap_width = 256
    heatmap_height = 128
    heatmap_channels = 3
    tile_extent = 32
    clip = 12
    npy_file_path = tmp_path / "test_edge_clipping_heatmap.npy"

    # keep workload reasonable for CI
    num_batches = 4
    B = 8

    assembler = EdgeClippingMultiscaleHeatmapAssembler(
        clip_top=clip,
        clip_bottom=clip,
        clip_left=clip,
        clip_right=clip,
        heatmap_width=heatmap_width,
        heatmap_height=heatmap_height,
        heatmap_channels=heatmap_channels,
        heatmap_npy_fp=npy_file_path,
    )

    # create dummy data: ones so sums are predictable
    example_tile_batch = np.ones((B, heatmap_channels, tile_extent, tile_extent), dtype=np.float32)

    increment = example_tile_batch[..., clip:tile_extent-clip, clip:tile_extent-clip].sum()
    expected_increment = B * heatmap_channels * (tile_extent - 2 * clip) * (tile_extent - 2 * clip)
    assert increment == expected_increment, (
        f"Checksum mismatch in example tile batch: Should be {expected_increment}, is {increment}"
    )

    # add the tiles randomly to cover the heatmap
    for i in range(num_batches):
        max_x = max(1, heatmap_width - (tile_extent - clip))
        max_y = max(1, heatmap_height - (tile_extent - clip))
        x = (np.random.rand(B,) * max_x).astype(int)
        y = (np.random.rand(B,) * max_y).astype(int)

        assembler.update_batch_torch(example_tile_batch, x, y)

        # After each update the accumulator sum should grow by `increment`
        acc_sum = assembler.heatmap_accumulator.sum()
        assert acc_sum == (i + 1) * expected_increment, (
            f"Checksum mismatch in heatmap accumulator after update {i+1}. "
            f"Is {acc_sum}, but expected {(i+1) * expected_increment}"
        )

    total_expected = num_batches * expected_increment
    assert assembler.heatmap_accumulator.sum() == total_expected, (
        f"Checksum mismatch in heatmap accumulator after updates. Is {assembler.heatmap_accumulator.sum()}, "
        f"but expected {total_expected}"
    )

    assembled_heatmap, overlap_counter = assembler.finalize()

    assert assembled_heatmap.shape == (heatmap_channels, heatmap_height, heatmap_width), (
        f"Assembled heatmap has incorrect shape: {assembled_heatmap.shape}, shall be "
        f"{(heatmap_channels, heatmap_height, heatmap_width)}"
    )

    assert overlap_counter.shape == (heatmap_height, heatmap_width), (
        f"Overlap counter has incorrect shape: {overlap_counter.shape}, shall be "
        f"{(heatmap_height, heatmap_width)}"
    )

    # Checksum of overlap counter equals number of contributing pixels (each add increments by 1)
    expected_overlap_sum = num_batches * B * (tile_extent - 2 * clip) * (tile_extent - 2 * clip)
    assert overlap_counter.sum() == expected_overlap_sum, (
        f"Checksum mismatch in overlap counter after finalization. Is {overlap_counter.sum()}, "
        f"but expected {expected_overlap_sum}"
    )

    # After normalization, max value should be 1.0 because we added ones only
    assert assembled_heatmap.max() == 1.0, (
        f"Assembled heatmap values should be normalized to 1.0 after finalization: {assembled_heatmap.max()}"
    )


def test_heatmap_assembler_basic(tmp_path):
    """Test for the original MultichannelHeatmapAssembler (no clipping).

    Uses tmp_path to store a temporary memmap .npy and keeps sizes small for CI.
    """
    from explainability.precomputing import MultichannelHeatmapAssembler

    np.random.seed(1)

    heatmap_width = 256 + 7
    heatmap_height = 128 + 15
    heatmap_channels = 3
    tile_extent = 32
    npy_file_path = tmp_path / "test_heatmap.npy"

    # smaller workload for CI
    num_batches = 4
    B = 8

    assembler = MultichannelHeatmapAssembler(
        heatmap_width=heatmap_width,
        heatmap_height=heatmap_height,
        heatmap_channels=heatmap_channels,
        heatmap_npy_fp=npy_file_path,
    )

    example_tile_batch = np.ones((B, heatmap_channels, tile_extent, tile_extent), dtype=np.float32)
    increment = example_tile_batch.sum()
    expected_increment = B * heatmap_channels * tile_extent * tile_extent
    assert increment == expected_increment

    for i in range(num_batches):
        max_x = max(1, heatmap_width - tile_extent)
        max_y = max(1, heatmap_height - tile_extent)
        x = (np.random.rand(B,) * max_x).astype(int)
        y = (np.random.rand(B,) * max_y).astype(int)

        assembler.update_batch_torch(example_tile_batch, x, y)

        acc_sum = assembler.heatmap_accumulator.sum()
        assert acc_sum == (i + 1) * expected_increment, (
            f"Checksum mismatch in heatmap accumulator after update {i+1}. Is {acc_sum}, but expected {(i+1)*expected_increment}"
        )

    total_expected = num_batches * expected_increment
    assert assembler.heatmap_accumulator.sum() == total_expected

    assembled_heatmap, overlap_counter = assembler.finalize()

    assert assembled_heatmap.shape == (heatmap_channels, heatmap_height, heatmap_width)
    assert overlap_counter.shape == (heatmap_height, heatmap_width)
    expected_overlap_sum = num_batches * B * tile_extent * tile_extent
    assert overlap_counter.sum() == expected_overlap_sum
    assert assembled_heatmap.max() == 1.0
