import concurrent.futures
import os
import sys
from pathlib import Path

CHUNK_SIZE = 64


def run_count_diff(file1, file2):
    cmd = f"./count_diff {file1} {file2} {CHUNK_SIZE}"
    output = os.popen(cmd).read().split()[0]
    return int(output)


def main(dumpdir):
    dump_files = list(sorted(Path(dumpdir).iterdir(), key=os.path.getmtime))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(run_count_diff, dump_files[:-1], dump_files[1:])

        for i, count in enumerate(results):
            print(f"{i},{i+1},{count}")


if __name__ == '__main__':
    # Check if folder path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Please provide folder path as a command-line argument")
        sys.exit(1)
    if len(sys.argv) == 3:
        CHUNK_SIZE = int(sys.argv[2])
    main(sys.argv[1])
