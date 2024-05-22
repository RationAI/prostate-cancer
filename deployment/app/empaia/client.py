import asyncio
from functools import cached_property
from types import TracebackType
from typing import Any

from aiohttp import ClientSession

from app.empaia.typing import WSIMask


class Client:
    def __init__(self, api_url: str, job_id: str, token: str) -> None:
        self.api_url = api_url
        self.job_id = job_id
        self.token = token

    @cached_property
    def _session(self) -> ClientSession:
        return ClientSession(
            self.api_url,
            headers={"Authorization": f"Bearer {self.token}"},
            raise_for_status=True,
        )

    async def __aenter__(self) -> "Client":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    def __reduce__(self) -> tuple[type["Client"], tuple[str, str, str]]:
        serialized_data = (self.api_url, self.job_id, self.token)
        return type(self), serialized_data

    def __del__(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._session.close())
            else:
                loop.run_until_complete(self._session.close())
        except RuntimeError:
            pass

    async def get_input(self, input_key: str) -> Any:
        response = await self._session.get(f"/v3/{self.job_id}/inputs/{input_key}")
        return await response.json()

    async def post_output(self, output_key: str, body: dict) -> Any:
        response = await self._session.post(
            f"/v3/{self.job_id}/outputs/{output_key}", json=body
        )
        return await response.json()

    async def get_region(
        self, wsi_id: str, level: int, x: int, y: int, width: int, height: int
    ) -> bytes:
        response = await self._session.get(
            f"/v3/{self.job_id}/regions/{wsi_id}/level/{level}/start/{x}/{y}/size/{width}/{height}"
        )
        return await response.read()

    async def put_tile(
        self, pixelmap_id: str, level: int, tile_x: int, tile_y: int, data: bytes
    ) -> None:
        await self._session.put(
            f"/v3/{self.job_id}/pixelmaps/{pixelmap_id}/level/{level}/position/{tile_x}/{tile_y}/data",
            data=data,
        )

    async def put_progress(self, progress: float) -> None:
        await self._session.put(
            f"/v3/{self.job_id}/progress", json={"progress": progress}
        )

    async def put_finalize(self) -> None:
        await self._session.put(f"/v3/{self.job_id}/finalize")

    async def put_failure(self, user_message: str) -> None:
        await self._session.put(
            f"/v3/{self.job_id}/failure", json={"user_message": user_message}
        )

    async def post_wsi_mask(self, wsi_id: str, path: str) -> WSIMask:
        response = await self._session.post(
            f"/v3/{self.job_id}/wsi_mask/{wsi_id}/create", json={"path": path}
        )
        return await response.json()

    async def close(self) -> None:
        if "_session" in self.__dict__:
            await self._session.close()
