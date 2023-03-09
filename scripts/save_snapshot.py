import asyncio
import os
from typing import Optional, Mapping

from qemu.qmp import QMPClient


async def qmp_execute(qmp: QMPClient, cmd: str, args: Optional[Mapping[str, object]] = None) -> dict:
    res = await qmp.execute(cmd, args)
    assert isinstance(res, dict)
    return res


async def main(port: int, iterations: int, epoch_ms: int, tag: str) -> None:
    qmp = QMPClient('nvram')
    await qmp.connect(('localhost', port))

    res = await qmp_execute(qmp, 'query-status')
    status = res['status']
    print(f"VM status: {status}")

    for i in range(iterations):
        cwd = os.getcwd()
        filename = f'{cwd}/{tag}-{i}.dump'
        print(f"Saving snapshot {i} to {filename}...")
        res = await qmp_execute(qmp, 'dump-guest-memory', {'paging': False, 'protocol': f'file:{filename}'})
        if res:
            raise RuntimeError("Failed to dump memory", res)
        if status != 'running':
            print("VM is not running, stopping.")
            break
        await asyncio.sleep(epoch_ms / 1000)
    print("Done.")
    await qmp.disconnect()


if __name__ == '__main__':
    # parse command line arguments
    # <port> <run-seconds> <epoch-ms> <tag>
    import argparse

    parser = argparse.ArgumentParser(description="Save a snapshot of the VM's memory.")
    parser.add_argument('port', type=str, default='4444', help='The port to connect to.')
    parser.add_argument('iterations', type=int, help='The number of iterations to run.')
    parser.add_argument('epoch_ms', type=int, help='The number of milliseconds between snapshots.')
    parser.add_argument('tag', type=str, help='The name of the snapshot to save.')
    args = parser.parse_args()
    asyncio.run(main(
        port=args.port,
        iterations=args.iterations,
        epoch_ms=args.epoch_ms,
        tag=args.tag)
    )
