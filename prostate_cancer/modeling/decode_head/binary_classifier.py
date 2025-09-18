from torch import Tensor, nn


class BinaryClassifier(nn.Module):
    """This is a single neuron classifier for feature vectors.

    Note:
        Since CNN extractors provide features in 4 dimensional tensor, we use global pooling to flatten it.
        This is not neccessary for GigaPath features which are already flat.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.proj = nn.Linear(in_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        # CNN feature extractors
        if x.ndim == 4:  # (B, C, W, H)
            x = self.global_pool(x)  # (B, C, 1, 1)
            x = x.flatten(start_dim=-3, end_dim=-1)  # (B, C)
        # PGP already provides 2 dimensions
        elif x.ndim != 2:  # (B, C)
            raise ValueError(
                f"Expected input of shape (B, C) or (B, C, W, H), got {x.shape}"
            )

        x = self.dropout(x)
        x = self.proj(x)
        return x  # Return logits, not probability
