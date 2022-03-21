import networkx as nx

import bcode
import graph_utils as gu

BLOCKNAME = 'bcode_block'


def update_stackdepth(cfg):
    """The stack depth is supposed to be independent of path, so dijkstra on the undirected graph suffices
    (and may be too strong, since we don't need minimality).
    The `undirected` part is just because we want to work with unreachable code too.
    """
    bidi = gu.copy_to_bidirectional(cfg, weight='stack_effect')
    depths = nx.single_source_dijkstra_path_length(bidi, source=0, weight='stack_effect')
    nx.set_node_attributes(cfg, depths, 'stack_depth')
    return cfg


def make_bcode_block_cfg(instructions):
    dbs = {b.offset: b for b in instructions}
    edges = [(b.offset, dbs[j].offset, {'stack_effect': stack_effect})
             for b in dbs.values()
             for (j, stack_effect) in b.next_list() if dbs.get(j) is not None]
    cfg = nx.DiGraph(edges)
    nx.set_node_attributes(cfg, name='BCode', values=dbs)
    update_stackdepth(cfg)
    # each node will hold a block of dictionaries - bcode and stack_depth
    basic_block_cfg = gu.contract_chains(cfg, blockname=BLOCKNAME)
    return basic_block_cfg


def get_code_depth_pairs(data):
    return [(d['BCode'], d['stack_depth']) for d in data[BLOCKNAME]]


def print_graph(cfg):
    for b in sorted(cfg.nodes()):
        print(b, ':', cfg.nodes[b][BLOCKNAME])


def draw(g: nx.DiGraph):
    import matplotlib.pyplot as plt
    nx.draw_networkx(g, with_labels=True)
    plt.show()


def make_bcode_block_cfg_from_function(f):
    instructions = bcode.get_instructions(f)
    return make_bcode_block_cfg(instructions)


def test():
    import code_examples
    cfg = make_bcode_block_cfg_from_function(code_examples.CreatePlasmaCube)
    print_graph(cfg)


if __name__ == '__main__':
    test()
