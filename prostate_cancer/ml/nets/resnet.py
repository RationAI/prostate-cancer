# Copyright (c) The RationAI team.

import mlflow
import torch
from torch import nn
from torchvision.models import resnet50


class ResNet50:
    def __new__(cls, weights=None, model_uri=None) -> torch.nn.Module:
        if weights is not None and model_uri is not None:
            raise ValueError("Only one of weights or model_uri can be specified")
        if model_uri:
            donor = mlflow.pytorch.load_model(model_uri)
        else:
            donor = resnet50(weights=weights)
            donor = nn.Sequential(*list(donor.children())[:-2])
        return donor
