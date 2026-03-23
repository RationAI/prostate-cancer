from torch import Tensor, nn


class BinaryClassifier(nn.Module):
    """Universal binary classifier head.

    Supports:
    - CNN features: (B, C, H, W)
    - Flat features: (B, C)
    - Transformer features (e.g. ViT): (B, N, C)
    """

    def __init__(
        self,
        in_features: int,
        dropout: float = 0.5,
        pooling: str = "cls",  # "cls" or "mean" for transformers
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.pooling = pooling

        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(in_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        # CNN case: (B, C, H, W)
        if x.ndim == 4:
            x = self.global_pool(x)          # (B, C, 1, 1)
            x = x.flatten(1)                # (B, C)

        # Transformer case: (B, N, C)
        elif x.ndim == 3:
            if self.pooling == "cls":
                x = x[:, 0]                # CLS token
            elif self.pooling == "mean":
                x = x.mean(dim=1)          # mean pooling
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

        # Already flat: (B, C)
        elif x.ndim != 2:
            raise ValueError(
                f"Expected input of shape (B, C), (B, N, C), or (B, C, H, W), got {x.shape}"
            )

        # Final classification
        x = self.dropout(x)
        x = self.proj(x)

        return x
