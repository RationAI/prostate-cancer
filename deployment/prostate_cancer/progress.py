import shutil

import aiohttp
import ray


@ray.remote(num_cpus=0.1)
class Progress:
    progress_log_interval = 0.01

    def __init__(
        self,
        api_url: str,
        job_id: str,
        headers: dict[str, str],
        total: int,
        checkpoint_dir: str,
    ) -> None:
        self.job_id = job_id
        self.total = total
        self.checkpoint_dir = checkpoint_dir

        self._session = aiohttp.ClientSession(api_url, headers=headers)
        self._finished = 0
        self._last_logged_progress = 0

    async def update(self, current: int) -> None:
        self._finished += current
        progress = self._finished / self.total
        if (
            progress - self._last_logged_progress >= self.progress_log_interval
            and self._session is not None
        ):
            await self._session.put(
                f"/v3/{self.job_id}/progress", json={"progress": progress}
            )
            self._last_logged_progress = progress

    async def finalize(self) -> None:
        await self._session.put(f"/v3/{self.job_id}/progress", json={"progress": 1})
        await self._session.close()
        shutil.rmtree(self.checkpoint_dir)
