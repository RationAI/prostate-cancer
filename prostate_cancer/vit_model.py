from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn
from transformers import ViTModel

from prostate_cancer.base_model import ProstateCancerModel


class ViTProstateModel(ProstateCancerModel):
    def __init__(
        self, backbone: ViTModel, decode_head: nn.Module, lr: float, tl_threshold: float
    ) -> None:
        super().__init__(lr=lr, tl_threshold=tl_threshold)
        self.backbone = backbone
        self.decode_head = decode_head

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x).last_hidden_state
        logits = self.decode_head(features)
        return logits

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = False, assign: bool = False
    ) -> Any:
        return super().load_state_dict(
            state_dict, strict=False, assign=assign
        )  # we have one ViT model containing pooler (unused, but present in the checkpoint)
