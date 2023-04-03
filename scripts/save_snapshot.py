"""
A script to save a snapshot of the VM's memory:
$iterations times do:
    1. pause the VM
    2. dump memory to two hard links "{i}.a.dump" and "{i-1}.b.dump"
    3. run concurrently count_diff on {i-1}.a.dump and {i-1}.b.dump (both will be removed by count_diff)
    4. continue the VM
    5. sleep X milliseconds
    Finally, wait for all the concurrently running processes to finish.

"server" does the same, but reads the commands from a TCP socket and does not sleep.
"""
import asyncio
import os
import pathlib
import socket
import struct
import subprocess
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Iterator

from qmp_client import SimpleQmpClient


class Server:
    sleep_duration_ms = 0
    tag: str
    def __next__(self) -> int: raise NotImplementedError

    def __iter__(self) -> Iterator[int]:
        return self


class SimpleTcpServer(Server):
    def __init__(self, port: int):
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('localhost', port))
        self.server_socket.listen(1)
        print(f"Listening on port {port}", file=sys.stderr)
        self.client_socket, client_address = self.server_socket.accept()
        self.tag = self.client_socket.recv(1024).decode('utf8')
        print(f"Tag: {self.tag}", file=sys.stderr)
        if not self.tag.replace('_', '').isalnum():
            raise ValueError("Invalid tag", self.tag)

    def __enter__(self) -> 'SimpleTcpServer':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client_socket.close()
        self.server_socket.close()

    def __next__(self) -> int:
        raw_index = self.client_socket.recv(8)
        if not raw_index:
            raise StopIteration
        index = struct.unpack('Q', raw_index)[0]
        print(f"Received: {index!r}", end='\r', flush=True, file=sys.stderr)
        return int(index)


class IteratorServer(Server):
    def __init__(self, iterations: int, sleep_duration_ms: int, tag: str):
        self.iterations = iterations
        self.sleep_duration_ms = sleep_duration_ms
        self.tag = tag
        self.i = 0

    def __next__(self) -> int:
        self.i += 1
        if self.i > self.iterations:
            raise StopIteration
        return self.i


def count_diff(folder, i):
    result = subprocess.run(["./count_diff",
                             f"{folder}/{i}.a.dump",
                             f"{folder}/{i}.b.dump",
                             "64",
                             f"remove"
                             ],
                            capture_output=True)
    if result.returncode != 0:
        raise RuntimeError("Failed to run count_diff", result)
    return int(result.stdout.strip())


async def relay_qmp_dumps(qmp_port: int, server: Server) -> None:
    tag = server.tag
    folder = f'{pathlib.Path.cwd().as_posix()}/dumps/{tag}'
    os.makedirs(folder, exist_ok=True)
    async with SimpleQmpClient(qmp_port) as vm:
        await vm.dump(f'{folder}/0.a.dump')
        with ThreadPoolExecutor() as executor:
            ps: list[Future[int]] = []
            for index in server:
                async with vm.pause(server.sleep_duration_ms):
                    await vm.dump(f'{folder}/{index}.b.dump')
                    next_prev_file = f'{folder}/{index + 1}.a.dump'
                    os.link(f'{folder}/{index}.b.dump', next_prev_file)
                    p: Future[int] = executor.submit(count_diff, folder, index)
                    if p is not None:
                        ps.append(p)
            else:
                os.unlink(next_prev_file)
            with open(f'{folder}.csv', 'w') as f:
                for i, p in enumerate(ps):
                    print(f"{i},{p.result()}", file=f)
            os.rmdir(folder)
            print("Done.", file=sys.stderr)


def run_server(qmp_port: int, tcp_port: int):
    with SimpleTcpServer(tcp_port) as server:
        asyncio.run(relay_qmp_dumps(qmp_port, server))


def run_iterator(qmp_port: int, iterations: int, sleep_duration_ms: int, tag: str):
    assert iterations > 0
    server = IteratorServer(iterations, sleep_duration_ms, tag)
    asyncio.run(relay_qmp_dumps(qmp_port, server))


def main():
    parser = argparse.ArgumentParser(description="Save a snapshot of the VM's memory.")
    parser.add_argument('qmp_port', type=int, default=4444, help='The port in qemu to connect the QMP client to.')
    # subparsers: server and iterator
    subparsers = parser.add_subparsers()
    server_parser = subparsers.add_parser('server', help='Run as a server.')
    server_parser.add_argument('tcp_port', type=int, default=1234, help='TCP port to listen on.')
    server_parser.set_defaults(func=run_server)

    iterator_parser = subparsers.add_parser('iterator', help='Run as an iterator.')
    iterator_parser.add_argument('iterations', type=int, help='The number of iterations to run.')
    iterator_parser.add_argument('sleep_duration_ms', type=int, help='The number of milliseconds between snapshots.')
    iterator_parser.add_argument('tag', type=str, help='Save as ./dumps/[tag].csv.')
    iterator_parser.set_defaults(func=run_iterator)

    args = parser.parse_args()
    func = args.func
    del args.func
    func(**vars(args))


if __name__ == '__main__':
    main()
