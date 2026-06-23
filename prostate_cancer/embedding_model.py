from torch import Tensor

from prostate_cancer.base_model import ProstateCancerModel
from prostate_cancer.modeling.decode_head import BinaryEmbeddingClassifier


class EmbeddingProstateModel(ProstateCancerModel):
    def __init__(
        self, decode_head: BinaryEmbeddingClassifier, lr: float, tl_threshold: float
    ) -> None:
        super().__init__(lr=lr, tl_threshold=tl_threshold)
        self.decode_head = decode_head

    def forward(self, x: Tensor) -> Tensor:
        logits = self.decode_head(x)
        return logits
