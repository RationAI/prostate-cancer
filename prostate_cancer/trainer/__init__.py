# Copyright (c) The RationAI team.

import lightning


class Trainer(lightning.Trainer):
    def __init__(self, *args, **kwargs):
        """A wrapper around the lightning.Trainer class that allows for callbacks to be passed as a dict.

        This allows simpler callbacks configuration using Hydra
        """
        if "callbacks" in kwargs and isinstance(kwargs["callbacks"], dict):
            kwargs["callbacks"] = list(kwargs["callbacks"].values())
        super().__init__(*args, **kwargs)
