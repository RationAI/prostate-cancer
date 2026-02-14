import torch
import torch.nn.functional as F
from rationai.mlkit.metrics.aggregators import HeatmapAggregator
from torch import Tensor


class SpanAggregator(HeatmapAggregator):
    """Binary aggregator detecting existence of cell in heatmap such that in at least one direction (horizontal vertical) there is a span of at least k positive cells."""

    def __init__(
        self,
        extent_tile: int,
        stride_tile: int,
        k: int,
        cell_threshold: float,
    ) -> None:
        super().__init__(extent_tile, stride_tile)
        self.k = k
        self.cell_threshold = cell_threshold

    def compute(self) -> tuple[Tensor, Tensor]:
        binary_heatmap = self._get_heatmap() >= self.cell_threshold
        h, w = binary_heatmap.shape
        batch_heatmap = binary_heatmap.unsqueeze(0).unsqueeze(0).float()  # (1,1,h,w)

        # Horizontal windows (erosion with horizontal SE)
        h_kernel = torch.ones((1, 1, 1, self.k), device=binary_heatmap.device)
        h_sum = F.conv2d(batch_heatmap, h_kernel)  # (1,1,h,w-k+1)
        h_runs = h_sum == self.k  # spans at least k cells

        h_mask = torch.zeros_like(binary_heatmap)

        # perform OR for each offset (anchor of SE)
        for t in range(self.k):
            h_mask[:, t : t + w - self.k + 1] |= h_runs[0, 0]

        # Vertical windows (erosion with vertical SE)
        v_kernel = torch.ones((1, 1, self.k, 1), device=binary_heatmap.device)
        v_sum = F.conv2d(batch_heatmap, v_kernel)  # (1,1,h-k+1,w)
        v_runs = v_sum == self.k

        v_mask = torch.zeros_like(binary_heatmap)
        for t in range(self.k):
            v_mask[t : t + h - self.k + 1, :] |= v_runs[0, 0]

        # is there a position where any condition is met ? (OR)
        pred = (h_mask | v_mask).any()
        target = torch.cat(self.targets).max()

        return pred, target
