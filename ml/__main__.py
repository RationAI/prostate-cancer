import logging
import resource
from random import randint

import hydra
import torch
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import Trainer, autolog
from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger

from ml._mlflow_compat import apply_mlflow_compat_patch


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

_logger = logging.getLogger(__name__)


def _raise_open_file_limit(target: int = 8192) -> None:
    """Raise the process' open-file soft limit.

    Each DataLoader worker keeps one open OpenSlide handle per slide it
    touches (see CachedOpenSlideTilesDataset), so a training run can hold
    thousands of file descriptors open at once. The deployment's default
    ulimit (often 1024) may not cover that, so raise it here rather than
    relying on external environment setup.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = target if hard == resource.RLIM_INFINITY else min(target, hard)

    if new_soft <= soft:
        return

    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))

    if new_soft < target:
        _logger.warning(
            "Open-file hard limit (%d) is below the requested %d; raised soft "
            "limit to %d instead. Too many open slide handles may still hit "
            "'Too many open files'.",
            hard,
            target,
            new_soft,
        )


@hydra.main(config_path="../configs", config_name="ml", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    apply_mlflow_compat_patch()
    _raise_open_file_limit()
    seed_everything(config.seed, workers=True)

    torch.set_float32_matmul_precision(precision="medium")

    data = hydra.utils.instantiate(
        config.datamodule,
        _recursive_=False,  # to avoid instantiating all the datasets
    )

    model = hydra.utils.instantiate(
        config.model
    )  # Model target is required in the config file

    trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer, logger=logger)

    # Run the trainer in the specified mode
    getattr(trainer, config.mode)(model, datamodule=data, ckpt_path=config.checkpoint)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
