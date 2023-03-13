import argparse
import concurrent.futures
import os
from pathlib import Path

CHUNK_SIZE = 64


def run_count_diff(file1, file2):
    cmd = f"./count_diff {file1} {file2} {CHUNK_SIZE}"
    output = os.popen(cmd).read().split()[0]
    return int(output)


def main(dumpdir, chunk_size, max_workers):
    dump_files = list(sorted(Path(dumpdir).iterdir(), key=os.path.getmtime))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(run_count_diff, dump_files[:-1], dump_files[1:])

        for i, count in enumerate(results):
            print(f"{i},{i+1},{count}")


if __name__ == '__main__':
    # Check if folder path is provided as a command-line argument
    parser = argparse.ArgumentParser(description="Compute count_diff for pairs of dump files")
    parser.add_argument("dump_dir", type=str, help="Directory containing the dump files")
    parser.add_argument("-c", "--chunk_size", type=int, default=64, help="Chunk size for comparison")
    parser.add_argument("-w", "--max_workers", type=int, default=10, help="Maximum number of worker threads")
    args = parser.parse_args()

    CHUNK_SIZE = args.chunk_size
    main(args.dump_dir, args.chunk_size, args.max_workers)
