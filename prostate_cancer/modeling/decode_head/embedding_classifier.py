from torch import Tensor

from prostate_cancer.modeling.decode_head.binary_classifier import BinaryClassifier

class BinaryEmbeddingClassifier(BinaryClassifier):
    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        x = self.dropout(x)
        x = self.proj(x)
        return x
