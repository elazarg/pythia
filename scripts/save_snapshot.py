import asyncio
import os
import pathlib
import subprocess
import sys
from typing import Optional, Mapping

from qemu.qmp import QMPClient
from qemu.qmp.protocol import ConnectError


async def qmp_execute(qmp: QMPClient, cmd: str, args: Optional[Mapping[str, object]] = None) -> dict:
    res = await qmp.execute(cmd, args)
    assert isinstance(res, dict)
    return res


def run_count_diff(file1, file2):
    # Run in a detached process and write the output to a file
    cmd = f"./count_diff {file1} {file2} {64}"
    subprocess.call(cmd, shell=True)


def system(cmd):
    print(cmd, file=sys.stderr)
    os.system(cmd)


async def main(port: int, iterations: int, epoch_ms: int, tag: str) -> None:
    cwd = pathlib.Path.cwd()
    folder = cwd / 'dumps' / tag
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

    prev_filename = "/dev/null"
    for i in range(iterations):
        filename = (folder / f'{i}.dump').as_posix()
        print(f"Saving snapshot {i} to {filename}...", file=sys.stderr)
        res = await qmp_execute(qmp, 'dump-guest-memory', {'paging': False, 'protocol': f'file:{filename}'})
        if res:
            raise RuntimeError("Failed to dump memory", res)
        if status != 'running':
            print("VM is not running, stopping.", file=sys.stderr)
            break
        if i > 0:
            outfile = (folder / f'{i}.diff').as_posix()
            system(f"printf '{i},' > {outfile} && "
                   f"./count_diff {prev_filename} {filename} {64} >> {outfile} && "
                   f"rm -f {prev_filename} &")
        await asyncio.sleep(epoch_ms / 1000)
        prev_filename = filename
    tag = folder.as_posix()
    system(f"rm -f {prev_filename}")
    system(f"sort -t=',' -g {tag}/*.diff > {tag}.csv && "
           f"rm -f {tag}/* && "
           f"rmdir {folder}")
    # TODO: wait for all subprocesses to finish
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
