from typing import Iterable

import bcode
import graph_utils as gu

Cfg = gu.Cfg[bcode.BCode]


def calculate_stack_depth(cfg: Cfg) -> dict[int, int]:
    """The stack depth is supposed to be independent of path, so dijkstra on the undirected graph suffices
    (and may be too strong, since we don't need minimality).
    We do it bidirectionally because we want to work with unreachable code too.
    """
    res: dict[int, int] = {}
    backwards_cfg = cfg.reverse(copy=True)
    for label in cfg.nodes:
        if cfg[label][-1].is_return:
            # add 1 since return also pops
            dijkstra = gu.single_source_dijkstra_path_length(backwards_cfg, source=label, weight='stack_effect')
            res.update({k: v+1 for k, v in dijkstra.items()})
    res.update(gu.single_source_dijkstra_path_length(cfg, source=0, weight='stack_effect'))
    return res


def make_bcode_block_cfg(instructions: Iterable[bcode.BCode]) -> tuple[dict[int, int], Cfg]:
    instructions = list(instructions)
    for ins in instructions:
        print(ins.offset, ins)
    dbs = {ins.offset: ins for ins in instructions}
    edges = [(b.offset, dbs[j].offset, {'stack_effect': stack_effect})
             for b in dbs.values()
             for (j, stack_effect) in b.next_list() if dbs.get(j) is not None]
    cfg = gu.Cfg(edges, blocks={k: [v] for k, v in dbs.items()})
    depths = calculate_stack_depth(cfg)
    # each node will hold a block of dictionaries - bcode and stack_depth
    return depths, cfg


def make_bcode_block_cfg_from_function(f) -> tuple[dict[int, int], Cfg]:
    instructions = bcode.get_instructions(f)
    return make_bcode_block_cfg(instructions)


def test() -> None:
    import code_examples
    _, cfg = make_bcode_block_cfg_from_function(code_examples.CreatePlasmaCube)
    gu.simplify_cfg(cfg).print_graph()


if __name__ == '__main__':
    test()
