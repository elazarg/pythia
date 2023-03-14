import asyncio
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Mapping

from qemu.qmp import QMPClient
from qemu.qmp.protocol import ConnectError


async def qmp_execute(qmp: QMPClient, cmd: str, args: Optional[Mapping[str, object]] = None) -> dict:
    res = await qmp.execute(cmd, args)
    assert isinstance(res, dict)
    return res


def count_diff(folder, i):
    result = subprocess.run(["./count_diff",
                             f"{folder}/{i}.a.dump",
                             f"{folder}/{i}.b.dump",
                             f"{64}",
                             f"remove"],
                            capture_output=True)
    if result.returncode != 0:
        raise RuntimeError("Failed to run count_diff", result)
    output = int(result.stdout.strip())
    return output


async def main(port: int, iterations: int, epoch_ms: int, tag: str) -> None:
    cwd = pathlib.Path.cwd().as_posix()
    folder = f'{cwd}/dumps/{tag}'
    os.makedirs(folder, exist_ok=True)

    qmp = QMPClient('nvram')
    try:
        await qmp.connect(('localhost', port))
    except ConnectError:
        print(f"Failed to connect to QMP server.", file=sys.stderr)
        print(f"Check that the VM is running and listens at port {port}.", file=sys.stderr)
        sys.exit(1)
    res = await qmp_execute(qmp, 'query-status')
    status = res['status']
    print(f"VM status: {status}", file=sys.stderr)
    with ThreadPoolExecutor() as executor:
        ps = []
        for i in range(iterations):

            await qmp_execute(qmp, 'stop')
            filename = f'{folder}/{i}.a.dump'
            print(f"Step {i}", end='\r', file=sys.stderr)

            res = await qmp_execute(qmp, 'dump-guest-memory', {'paging': False, 'protocol': f'file:{filename}'})
            if res:
                raise RuntimeError("Failed to dump memory", res)
            if i > 0:
                if i == iterations - 1:
                    os.rename(filename, f"{folder}/{i-1}.b.dump")
                else:
                    os.link(filename, f"{folder}/{i-1}.b.dump")
                p = executor.submit(count_diff, folder, i - 1)
                ps.append(p)
            await qmp_execute(qmp, 'cont')

            await asyncio.sleep(epoch_ms / 1000)
        with open(f'{folder}.csv', 'w') as f:
            for i, p in enumerate(ps):
                print(f"{i},{p.result()}", file=f)
    os.rmdir(folder)
    print("Done.", file=sys.stderr)
    await qmp.disconnect()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Save a snapshot of the VM's memory.")
    parser.add_argument('port', type=str, default='4444', help='The port to connect to.')
    parser.add_argument('iterations', type=int, help='The number of iterations to run.')
    parser.add_argument('epoch_ms', type=int, help='The number of milliseconds between snapshots.')
    parser.add_argument('tag', type=str, help='Save as ./dumps/[tag].csv.')
    args = parser.parse_args()
    asyncio.run(main(
        port=args.port,
        iterations=args.iterations,
        epoch_ms=args.epoch_ms,
        tag=args.tag)
    )
