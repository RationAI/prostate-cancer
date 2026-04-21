from torch import Tensor

from prostate_cancer.modeling.decode_head.binary_classifier import BinaryClassifier


class BinaryEmbeddingClassifier(BinaryClassifier):
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {x.ndim}D")

        x = self.dropout(x)
        x = self.proj(x)
        return x
