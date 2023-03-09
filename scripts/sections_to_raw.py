"""Takes the dump images created by dump-guest-memory command
and split it to raw files.
"""
import sys

from elftools.elf.elffile import ELFFile  # type: ignore


def main(filename):
    with ELFFile.load_from_path(filename) as elf:
        for i, segment in enumerate(elf.iter_segments()):
            if i == 2:
                assert segment.header.p_type == 'PT_LOAD'
                data = segment.data()
                break
    with open(f'{filename}-{i}.raw', 'wb') as f:
        assert isinstance(data, bytes)
        f.write(data)


if __name__ == '__main__':
    main(sys.argv[1])
