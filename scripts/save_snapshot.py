import asyncio
import os
import pathlib
import subprocess
import sys
from typing import Optional, Mapping

from qemu.qmp import QMPClient
from qemu.qmp.protocol import ConnectError

import datetime


async def qmp_execute(qmp: QMPClient, cmd: str, args: Optional[Mapping[str, object]] = None) -> dict:
    res = await qmp.execute(cmd, args)
    assert isinstance(res, dict)
    return res


def system(cmd):
    # print(cmd, file=sys.stderr)
    os.system(cmd)


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

    prev_filename = "/dev/null"
    paths = [(f'{folder}/{i}.dump', f'{folder}/{i}.diff', f'{folder}/{i}.diff.tmp', f'{folder}/{i}.link')
             for i in range(iterations)]
    cmds = [f"ln {filename} {myfilename} && "
            f"printf '{i},' > {temp_diff} && "
            f"./count_diff {prev_filename} {myfilename} {64} >> {temp_diff} && "
            f"rm -f {prev_filename} {myfilename} && "
            f"mv {temp_diff} {diff_file} &"
            for i, (filename, diff_file, temp_diff, myfilename) in enumerate(paths)]
    for i in range(iterations):
        filename, diff_file, temp_diff, myfilename = paths[i]
        # print(f"Saving snapshot {i} to {filenames}...", file=sys.stderr)
        res = await qmp_execute(qmp, 'dump-guest-memory', {'paging': False, 'protocol': f'file:{filename}'})
        save_time = datetime.datetime.now()
        if res:
            raise RuntimeError("Failed to dump memory", res)
        if status != 'running':
            print("VM is not running, stopping.", file=sys.stderr)
            break
        if i > 0:
            os.system(cmds[i])
        prev_filename = filename
        passed = datetime.datetime.now() - save_time
        print("time passed:", passed.microseconds)
        await asyncio.sleep((epoch_ms - passed.microseconds) / 1000)
    for _, diff_file, _, _ in paths[1:]:
        system(f"until [ -f {diff_file} ]; do sleep 1; done")
    system(f"rm -f {prev_filename}")
    system(f"sort -t, -g {folder}/*.diff > {folder}.csv && "
           f"rm -f {folder}/*.diff && "
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
