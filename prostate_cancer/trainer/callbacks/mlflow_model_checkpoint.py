# Copyright (c) The RationAI team.

from pathlib import Path

import mlflow
from lightning.pytorch.callbacks import ModelCheckpoint


class MLFlowModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath):
        net = trainer.model.model
        artifact_path = f"model/{type(self).__name__}/{Path(filepath).stem}"
        mlflow.pytorch.log_model(
            net,
            artifact_path,
            pip_requirements=["torch==2.0.1+cu118", "torchvision==0.15.2+cu118"],
        )
