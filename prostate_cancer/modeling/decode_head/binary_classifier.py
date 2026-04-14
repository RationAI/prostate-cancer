from torch import Tensor, nn


class BinaryClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(in_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
