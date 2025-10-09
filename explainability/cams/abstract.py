import torch
import torch.nn as nn
from jaxtyping import Float


# from explainability.typing import Cams, Embeddings, EmbeddingsLocs


# class AbstractCAMHook:
#     """Abstract class for CAM hooks.

#     A CAM hook is a class that can be used to compute CAMs for a given model and layer.
#     """

#     def __init__(
#         self,
#         model,
#         target_layer_name: str,
#         method_name: str,
#         save_positive_activations: bool = False,
#     ):
#         self.model = model
#         self.target_layer_name = target_layer_name
#         self.method_name = method_name
#         self.save_positive_activations = save_positive_activations
#         self._forward_handle = None
#         self._activations = None  # [B,C,H,W]
#         self._cams_history: list[Cams] = []
#         self._embeddings: list[Embeddings] = []
#         self._embeddings_locations: list[EmbeddingsLocs] = []

#     def detach(self):
#         if self._forward_handle is not None:
#             self._forward_handle.remove()
#             self._forward_handle = None
#         self._activations = None

#     def clear(self):
#         self._cams_history.clear()

#     def last_cams(self):
#         if not self._cams_history:
#             return None
#         return self._cams_history[-1]

#     def last_embeddings(self) -> Embeddings:
#         assert self._embeddings, "No embeddings saved"
#         return self._embeddings[-1]

#     def last_embeddings_locations(self) -> EmbeddingsLocs:
#         assert self._embeddings_locations, "No embeddings locations saved"
#         return self._embeddings_locations[-1]

#     @torch.no_grad()
#     def _compute_cams(self, grad):
#         pass

#     @torch.no_grad()
#     def _save_embeddings(self, cams):
#         # For every spatial (i, j) where the CAM is positive, save the C-dim vector
#         # of activations across channels at that location, along with (b, i, j).
#         positive_indices = torch.nonzero(
#             cams > 0, as_tuple=False
#         )  # [K, 3] -> (b, i, j)

#         # if there are no positive indices, return
#         if positive_indices.numel() == 0:
#             return

#         b_idx = positive_indices[:, 0]
#         i_idx = positive_indices[:, 1]
#         j_idx = positive_indices[:, 2]
#         # Gather vectors across channels for each (b, i, j). Result: [K, C]
#         positive_vectors = self._activations[b_idx, :, i_idx, j_idx]
#         self._embeddings.append(positive_vectors.detach())
#         self._embeddings_locations.append(positive_indices.detach())

#     def _forward_hook(self, module, inputs, output):
#         self._activations = output  # keep tensor

#         def _save_grad_and_compute_cam(grad):
#             cams = self._compute_cams(grad)
#             self._cams_history.append(cams.detach())

#             if self.save_positive_activations:
#                 self._save_embeddings(cams)

#         output.register_hook(_save_grad_and_compute_cam)

#     def attach(self):
#         target_layer = None

#         for name, module in self.model.named_modules():
#             if name == self.target_layer_name:
#                 target_layer = module
#                 break

#         if target_layer is None:
#             raise ValueError(f"Layer '{self.target_layer_name}' not found in model")

#         self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
#         return self


def modify_ReLU_inplace(module, inplace=False):
    for layer in module._modules.values():
        if isinstance(layer, torch.nn.ReLU):
            layer.inplace = inplace
        modify_ReLU_inplace(layer, inplace=inplace)


def reshape_for_clustering(
    embeddings: Float[torch.Tensor, "B C H W"],
) -> tuple[Float[torch.Tensor, "BHW C"], tuple[int, int, int, int]]:
    B, C, H, W = embeddings.shape
    return embeddings.permute(0, 2, 3, 1).reshape(B * H * W, C), (B, C, H, W)


def reshape_from_clustering(
    embeddings: Float[torch.Tensor, "BHW C"], original_shape: tuple[int, int, int, int]
) -> Float[torch.Tensor, "B C H W"]:
    B, C, H, W = original_shape
    return embeddings.reshape(B, H, W, C).permute(0, 3, 1, 2)


class HookedModule(nn.Module):
    def __init__(self, model: nn.Module, layer_names: list[str]):
        """Wraps a model and hooks into specific layers to capture activations and gradients.

        Args:
            model (nn.Module): The original model to wrap.
            layer_names (list[str]): List of layer names (dot-separated) to hook.
        """
        super().__init__()
        self.model = model
        self.activations = {}
        self.gradients = {}
        self.layer_names = layer_names

        for name in layer_names:
            layer = self.model.get_submodule(name)
            layer.register_forward_hook(self._make_forward_hook(name))
            layer.register_full_backward_hook(self._make_backward_hook(name))

    def _make_forward_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()

        return hook

    def _make_backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()

        return hook

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_activations(self, name=None):
        if name is None:
            return self.activations
        return self.activations.get(name, None)

    def get_gradients(self, name=None):
        if name is None:
            return self.gradients
        return self.gradients.get(name, None)
