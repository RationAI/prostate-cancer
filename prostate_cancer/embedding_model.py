from torch import Tensor, nn

from prostate_cancer.base_model import ProstateCancerModel


class EmbeddingProstateModel(ProstateCancerModel):
    def __init__(self, decode_head: nn.Module, lr: float) -> None:
        super().__init__(lr=lr)
        self.decode_head = decode_head

    def forward(self, x: Tensor) -> Tensor:
        logits = self.decode_head(x)
        return logits
