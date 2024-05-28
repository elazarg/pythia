import asyncio
import sys
from contextlib import asynccontextmanager
from qemu.qmp import QMPClient
from qemu.qmp.protocol import ConnectError
from typing import Optional, Mapping, AsyncIterator


class SimpleQmpClient:
    def __init__(self, port):
        self.port = port
        self.qmp = QMPClient("nvram")

    async def __aenter__(self):
        try:
            await self.qmp.connect(("localhost", self.port))
            res = await self.execute("query-status")
            status = res["status"]
            print(f"VM status: {status}", file=sys.stderr)
            return self
        except ConnectError as ex:
            ex.add_note(f"Failed to connect to QMP server.")
            ex.add_note(
                f"Check that the VM is running and listens at port {self.port}."
            )
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.qmp.disconnect()

    async def execute(
        self, cmd: str, args: Optional[Mapping[str, object]] = None
    ) -> dict:
        res = await self.qmp.execute(cmd, args)
        assert isinstance(res, dict)
        return res

    @asynccontextmanager
    async def pause(self, sleep_duration_ms: int) -> AsyncIterator[None]:
        if not sleep_duration_ms:
            yield
            return
        await self.execute("stop")
        try:
            yield
        finally:
            await self.execute("cont")
        await asyncio.sleep(sleep_duration_ms / 1000)

    async def dump(self, filename) -> None:
        res = await self.execute(
            "dump-guest-memory", {"paging": False, "protocol": f"file:{filename}"}
        )
        if res:
            raise RuntimeError("Failed to dump memory", res)
