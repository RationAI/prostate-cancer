import json
import logging
from pathlib import Path
from typing import TypedDict

import ray

from app.empaia import Client

log = logging.getLogger(__name__)


class State(TypedDict):
    progress: float
    last_logged: float


@ray.remote(num_cpus=0.1)
class Progress:
    """Progress tracking for a job.

    Attributes:
        log_interval: The interval at which to log the progress.
    """

    log_interval = 0.01

    def __init__(
        self,
        client: Client,
        checkpoint_dir: Path,
    ) -> None:
        self.client = client
        self.checkpoint_dir = checkpoint_dir

        self._state_checkpoint = checkpoint_dir / "state.json"
        self._state = self._restore_state()

    async def update(self, current: float) -> None:
        """Update the progress of the job.

        Args:
            current (float): The current progress update, not the accumulated one.
        """
        self._state["progress"] = min(self._state["progress"] + current, 1)

        if (
            self._state["progress"] - self._state["last_logged"] >= self.log_interval
            and self.client is not None
        ):
            await self.client.put_progress(self._state["progress"])
            self._state["last_logged"] = self._state["progress"]
            log.info("Progress: %d%", self._state["progress"] * 100)

        self._make_checkpoint()

    async def finalize(self) -> None:
        await self.client.put_progress(1)
        await self.client.close()

    def _restore_state(self) -> State:
        if self._state_checkpoint.exists():
            with open(self._state_checkpoint, "r", encoding="utf-8") as f:
                return json.load(f)

        return {"progress": 0.0, "last_logged": 0.0}

    def _make_checkpoint(self) -> None:
        with open(self._state_checkpoint, "w", encoding="utf-8") as f:
            json.dump(self._state, f)
