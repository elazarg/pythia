from typing import List # :)  

import instruction as bc
from instruction import BCode
import utils
import operator as op
import networkx as nx

def split_to_basic_block(bs: List[BCode]) -> List[List[BCode]]:
    return list(utils.partition(bs, before=op.attrgetter('is_jump_target'),
                                     after=op.attrgetter('is_jump_source')))

def prepare(f: 'e.g. function'):
    ins = bc.get_instructions(f)
    return split_to_basic_block(ins)


def print_blocks(blocks):
    for block in blocks:
        print(block[-1].jump_targets, end=': ')
        print(*['({0.opname}, {0.argval}, {0.offset})'.format(x)
                for x in block])

def make_graph(blocks):
    dbs = {x[0].offset: x for x in blocks}
    return nx.DiGraph([(t[0].starts_line, dbs.get(j, (bc.END,))[0].starts_line)
                    for t in blocks for j in t[-1].next_list])

def draw(g: nx.DiGraph):
    import matplotlib.pyplot as plt
    nx.draw_networkx(g, with_labels=True)
    plt.show()
   
def test():
    blocks = prepare(utils.partition)
    g = make_graph(blocks)
    draw(g)


if __name__ == '__main__':   
    test()
    