"""
A script to save a snapshot of the VM's memory:
$iterations times do:
    1. Pause the VM
    2. Dump memory to two hard links "{i}.a.dump" and "{i-1}.b.dump"
    3. Run concurrently count_diff on {i-1}.a.dump and {i-1}.b.dump (count_diff will remove both)
    4. Continue the VM
    5. Sleep X milliseconds
    Finally, wait for all the concurrently running processes to finish.

"Server" does the same, but reads the commands from a TCP socket and does not sleep.
"""

import asyncio
import logging
import os
import socket
import struct
import argparse
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Iterator

from checkpoint.qmp_client import SimpleQmpClient
from checkpoint import persist

logging.basicConfig(level=logging.INFO)


class Server:
    sleep_duration_ms = 0
    tag: str

    def __next__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[int]:
        return self

    def finish(self) -> None:
        pass


class SimpleTcpServer(Server):
    port: int
    server_socket: socket.socket
    client_socket: socket.socket
    tag: str

    def __init__(self, port: int):
        self.port = port
        # We could make this UDP, but TCP is fine too
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("localhost", port))
        self.server_socket.listen(1)
        logging.info(f"Listening on port {port}")
        self.client_socket, client_address = self.server_socket.accept()
        self.tag = self.client_socket.recv(1024).decode("utf8")
        logging.info(f"Tag: {self.tag}")
        if not self.tag.replace("_", "").isalnum():
            raise ValueError("Invalid tag", self.tag)

    def __enter__(self) -> "SimpleTcpServer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client_socket.send(b"Closing")
        self.client_socket.close()
        self.server_socket.close()

    def __next__(self) -> int:
        self.client_socket.send(b"Ack prev")
        raw_index = self.client_socket.recv(8)
        if not raw_index:
            raise StopIteration
        index, *state = struct.unpack("Q", raw_index)
        logging.info(f"Received: {index!r}, {state!r}")
        return int(index)

    def finish(self) -> None:
        self.client_socket.send(b"Finish")


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


async def relay_qmp_results(qmp_port: int, server: Server) -> None:
    tag = server.tag
    assert tag.replace("_", "").isalnum()
    dumps_folder = persist.make_dumps_folder(tag)
    async with SimpleQmpClient(qmp_port) as vm:
        await vm.dump(f"{dumps_folder}/0.a.dump")
        with ThreadPoolExecutor() as executor:
            ps: list[Future[int]] = []
            for i, index in enumerate(server):
                async with vm.pause(server.sleep_duration_ms):
                    current_next_file = dumps_folder / f"{i}.b.dump"
                    next_prev_file = dumps_folder / f"{i + 1}.a.dump"
                    await vm.dump(current_next_file)
                    os.link(current_next_file, next_prev_file)
                    p: Future[int] = executor.submit(
                        persist.diff_vm_snapshot, dumps_folder, i
                    )
                    if p is not None:
                        ps.append(p)
            else:
                server.finish()
                os.unlink(next_prev_file)

            results_folder = persist.make_results_folder(tag)
            version_number = len(
                [x for x in results_folder.iterdir() if x.name.startswith("vm_")]
            )
            with open(results_folder / f"vm_{version_number}.tsv", "w") as f:
                for i, p in enumerate(ps):
                    print(i, p.result(), sep="\t", file=f, flush=True)

    os.rmdir(dumps_folder)
    logging.info("Done.")


def run_server(qmp_port: int, tcp_port: int):
    with SimpleTcpServer(tcp_port) as server:
        asyncio.run(relay_qmp_results(qmp_port, server))


def run_iterator(qmp_port: int, iterations: int, sleep_duration_ms: int, tag: str):
    assert iterations > 0
    server = IteratorServer(iterations, sleep_duration_ms, tag)
    asyncio.run(relay_qmp_results(qmp_port, server))


def main():
    parser = argparse.ArgumentParser(description="Save a snapshot of the VM's memory.")
    parser.add_argument(
        "--qmp_port",
        type=int,
        default=4444,
        help="The port in qemu to connect the QMP client to.",
    )
    # subparsers: server and iterator
    subparsers = parser.add_subparsers()
    server_parser = subparsers.add_parser("server", help="Run as a server.")
    server_parser.add_argument(
        "--tcp_port", type=int, default=1234, help="TCP port to listen on."
    )
    server_parser.set_defaults(func=run_server)

    iterator_parser = subparsers.add_parser("iterator", help="Run as an iterator.")
    iterator_parser.add_argument(
        "iterations", type=int, help="The number of iterations to run."
    )
    iterator_parser.add_argument(
        "sleep_duration_ms",
        type=int,
        help="The number of milliseconds between snapshots.",
    )
    iterator_parser.add_argument("tag", type=str, help="Save as ./results/[tag].csv.")
    iterator_parser.set_defaults(func=run_iterator)

    args = parser.parse_args()
    func = args.func
    del args.func
    func(**vars(args))


if __name__ == "__main__":
    main()
