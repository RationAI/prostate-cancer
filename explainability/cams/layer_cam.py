import torch

from explainability.cams.abstract import AbstractCAMHook


class LayerCAMBatchHook(AbstractCAMHook):
    """Batch-wise Layer-CAM hook.

    Computes alpha weights per location per channel using the gradient of the output with respect to the layer's input.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, method_name="layer_cam")

    def _compute_cams(self, grad):
        cams = torch.relu(grad) * torch.relu(self._activations)
        cams = cams.sum(dim=1)
        cams = torch.relu(cams)
        return cams
