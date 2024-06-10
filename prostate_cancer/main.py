# Copyright (c) The RationAI team.

import random

import hydra
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf


OmegaConf.register_new_resolver(
    "random_seed", lambda: random.randint(0, 2**31), use_cache=True
)


@hydra.main(config_path="../conf", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    seed_everything(config.seed)

    datamodule = hydra.utils.instantiate(config.datamodule, _convert_="partial")
    model = hydra.utils.instantiate(config.ml, _convert_="partial")

    trainer = hydra.utils.instantiate(config.trainer, _convert_="partial")
    getattr(trainer, config.stage)(model, datamodule=datamodule)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
