from prostate_cancer.base_model import ProstateCancerModel

from torch import Tensor
from torch import nn
from transformers import ViTModel

class ViTProstateModel(ProstateCancerModel):

    def __init__(self, backbone: ViTModel, decode_head: nn.Module, lr: float) -> None:
        super().__init__(lr=lr)
        self.backbone = backbone
        self.decode_head = decode_head

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x).last_hidden_state
        logits = self.decode_head(features)
        return logits
