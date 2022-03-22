from typing import Iterable

import networkx as nx

import bcode
import graph_utils as gu


def calculate_stack_depth(cfg: nx.DiGraph) -> dict[int, int]:
    """The stack depth is supposed to be independent of path, so dijkstra on the undirected graph suffices
    (and may be too strong, since we don't need minimality).
    We do it bidirectionally because we want to work with unreachable code too.
    """
    res: dict[int, int] = {}
    backwards_cfg = cfg.reverse(copy=True)
    for n in cfg.nodes:
        if cfg.nodes[n]['block'][-1].is_return:
            # add 1 since return also pops
            res.update({k: v+1 for k, v in nx.single_source_dijkstra_path_length(backwards_cfg, source=n, weight='stack_effect').items()})
    res.update(nx.single_source_dijkstra_path_length(cfg, source=0, weight='stack_effect'))
    return res


def make_bcode_block_cfg(instructions: Iterable[bcode.BCode]) -> tuple[dict[int, int], nx.DiGraph]:
    instructions = list(instructions)
    for ins in instructions:
        print(ins.offset, ins)
    dbs = {ins.offset: ins for ins in instructions}
    edges = [(b.offset, dbs[j].offset, {'stack_effect': stack_effect})
             for b in dbs.values()
             for (j, stack_effect) in b.next_list() if dbs.get(j) is not None]
    cfg = nx.DiGraph(edges)
    nx.set_node_attributes(cfg, name='block', values={k: [v] for k, v in dbs.items()})
    depths = calculate_stack_depth(cfg)
    # each node will hold a block of dictionaries - bcode and stack_depth
    return depths, cfg


def print_graph(cfg):
    for b in sorted(cfg.nodes()):
        print(b, ':', cfg.nodes[b]['block'])


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
