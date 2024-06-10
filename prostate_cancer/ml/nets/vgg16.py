# Copyright (c) The RationAI team.

import mlflow
import torch
from torch import Tensor, flatten, nn
from torchvision.models import vgg16


class VGG16Features:
    def __new__(cls, weights=None, model_uri=None) -> torch.nn.Module:
        if weights is not None and model_uri is not None:
            raise ValueError("Only one of weights or model_uri can be specified")
        if model_uri:
            donor = mlflow.pytorch.load_model(model_uri)
        else:
            donor = vgg16(weights=weights)
        return donor.features


class GMaxPool(torch.nn.Module):
    def forward(self, x):
        x = torch.nn.functional.adaptive_max_pool2d(x, output_size=1)
        return x.flatten(start_dim=-3, end_dim=-1)


class BinaryClassifier(torch.nn.Sequential):
    def __init__(
        self, dropout_probability: float = 0.5, input_dim: int = 512, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.add_module("dropout", torch.nn.Dropout(p=dropout_probability))
        self.add_module("dense", torch.nn.Linear(input_dim, 1))

        torch.nn.init.xavier_uniform_(self.dense.weight)
        torch.nn.init.zeros_(self.dense.bias)


class VGG16RegressionAdapter(nn.Module):
    """VGG16regressionAdapater.

    VGG16regressionAdapater consists of 4 parts:
    1) `pre_features`  - the first block of the conv layer that should reduce
                         the dimensionality of input images
    2) `fearues`       - well-known optionally pre-train VGG16 layers
    3) `post_features` - 2 more blocks of conv layers similar to those in VGG16 to shrink
                         the first layer in the `model` to a reasonable size
    4) `model`         - the last block is very similar to VGG16 fully-connected part,
                         with modification in size of the first and last layer

    The main reason for `pre_features` and `post_features` is that VGG16RegressionAdapter
    works with images that are several times larger than the original (224x224x3) images
    for which was VGG16 designed
    """

    def __init__(
        self,
        width: int,
        height: int,
        weights: str | None = None,
        model_uri: str | None = None,
    ) -> None:
        super().__init__()

        self.pre_features = nn.Sequential(
            # conv0
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.features = VGG16Features(weights=weights, model_uri=model_uri)

        self.post_features = nn.Sequential(
            # conv6
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            # conv7
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.model = nn.Sequential(
            nn.Linear(1024 * (width // 256) * (height // 256), 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1),
        )

    def forward(self, x: Tensor, boundaries: list[int] | None = None) -> Tensor:
        x = self.pre_features(x)
        x = self.features(x)
        x = self.post_features(x)

        x = flatten(x, start_dim=1)
        x = self.model(x)

        if boundaries is not None:
            x = torch.bucketize(x, torch.tensor(boundaries), right=True)

        return x
