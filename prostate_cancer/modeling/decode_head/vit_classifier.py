from torch import Tensor

from prostate_cancer.modeling.decode_head.binary_classifier import BinaryClassifier


class BinaryViTClassifier(BinaryClassifier):
    def __init__(self, in_features: int, pooling: str, dropout: float = 0.5) -> None:
        super().__init__(in_features=in_features, dropout=dropout)
        self.pooling = pooling

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got {x.ndim}D")

        if self.pooling == "cls":
            x = x[:, 0]
        elif self.pooling == "mean":
            x = x.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        x = self.dropout(x)
        x = self.proj(x)
        return x
