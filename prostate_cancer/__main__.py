import logging
from random import randint

import hydra
import torch
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import Trainer, autolog
from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger

from prostate_cancer.datamodule import DataModule
from prostate_cancer.log_title import log_checkpoint_title


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@hydra.main(config_path="../configs", config_name="ml", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    seed_everything(config.seed, workers=True)

    torch.set_float32_matmul_precision(precision="medium")

    data = hydra.utils.instantiate(
        config.datamodule,
        _recursive_=False,  # to avoid instantiating all the datasets
        _target_=DataModule,
    )
    model = hydra.utils.instantiate(
        config.model
    )  # Model target is required in the config file

    trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer, logger=logger)

    if config.checkpoint and isinstance(trainer.logger, MLFlowLogger):
        log_checkpoint_title(trainer.logger, config.checkpoint)

    # Run the trainer in the specified mode
    getattr(trainer, config.mode)(model, datamodule=data, ckpt_path=config.checkpoint)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
