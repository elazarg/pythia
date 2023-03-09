"""Takes the dump images (from a run of autoqmp)
and turn them into deltas.  The delta information is Python pickled.
"""
import shutil
import sys
import pickle
import os

from elftools.elf.elffile import ELFFile  # type: ignore


def load_section(filename):
    """
    Load an ELF section from ELF core file
    """
    data_array = []
    with ELFFile.load_from_path(filename) as elf:
        # print("num segments ",elf.num_segments())
        for segment in elf.iter_segments():
            if segment.header.p_type == 'PT_LOAD':
                shdr = segment.header
                # print("memsize={} \t\t offset={}".format(shdr.p_memsz, shdr.p_offset))
                data_array.append((segment.data(), shdr.p_memsz))

    assert data_array
    return (data_array, elf.num_segments())


def compare_segments(seg0, seg1):
    """
    Compare segments at 64B granularity. Return (diff-count, [(offset, old-CL-bytes, new-CL-bytes)]
    """
    if seg0[1] != seg1[1]:
        print("different size segments: {} {}\n".format(seg0[1], seg1[1]))
        return "bad seg size"
    segsize = seg0[1]
    data0 = seg0[0]
    data1 = seg1[0]
    changes = []

    for offset in range(0, segsize - 64, 64):
        take = slice(offset, offset + 64)
        if data0[take] != data1[take]:
            changes.append((offset, data0[take], data1[take]))

    return (len(changes), changes)


def main(name):
    with open(name + '.deltas', 'wb') as outfile:
        for i in range(0, 1000):

            print('Processing {} => {} ...'.format(i, i + 1))
            dump_file_name_a = "./" + name + "-{}.dump".format(i)
            dump_file_name_b = "./" + name + "-{}.dump".format(i + 1)

            if not os.path.exists(dump_file_name_b):
                print("Done")
                break

            (d0, d0segs) = load_section(dump_file_name_a)
            (d1, d1segs) = load_section(dump_file_name_b)
            # assert d0segs == d1segs == 4
            if i == 0:
                print('Found start: {}..'.format(i))
                shutil.copy(dump_file_name_a, dump_file_name_a + ".base")
            print(f'Found number of segments: {d0segs} {d1segs}')
            epoch_result = tuple([compare_segments(prev, current) for prev, current in zip(d0, d1)])
            print('Pickling result ({}, {})'.format(i, [count for count, diff in epoch_result]))
            pickle.dump((i, epoch_result), outfile)


if __name__ == '__main__':
    main(sys.argv[1])
