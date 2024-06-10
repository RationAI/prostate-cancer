# Copyright (c) The RationAI team.

import lightning
import mlflow
import torch


class SavedModel(lightning.LightningModule):
    """SavedModel is used primarily to load a stored ml from MLFlow artifact storage.

    The only required parameter is the MLFlow ml URI.
    """

    def __new__(cls, model_uri: str) -> torch.nn.Module:
        return mlflow.pytorch.load_model(model_uri)
