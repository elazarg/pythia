# Data flow analysis and stuff.

from __future__ import annotations

import typing

import networkx as nx

import graph_utils as gu
import tac

from tac_analysis_domain import AbstractDomain
from tac_analysis_liveness import single_block_liveness
from tac_analysis_constant import ConstantDomain


def make_tacblock_cfg(f, propagate_consts=True, liveness=True, propagate_assignments=True, simplify=True):
    cfg = tac.make_tacblock_cfg(f)
    if simplify:
        cfg = gu.simplify_cfg(cfg)
    if propagate_consts:
        analyze(cfg, ConstantDomain)
    if liveness:
        for n in sorted(cfg.nodes()):
            block = cfg.nodes[n]['block']
            block = list(single_block_liveness(block))
            block.reverse()
            cfg.nodes[n]['block'] = block
    return cfg


def print_block(n, block):
    print(n, ':')
    for ins in block:
        print('\t', ins)


def analyze(g: nx.DiGraph, Analysis: typing.Type[AbstractDomain]) -> None:
    start = 0
    nx.set_node_attributes(g, {v: Analysis.bottom() for v in g.nodes()}, 'post_inv')
    nx.set_node_attributes(g, {v: Analysis.bottom() for v in g.nodes()}, 'pre_inv')
    g.nodes[start]['pre_inv'] = Analysis.top()
    wl = {start}
    while wl:
        u = wl.pop()
        udata = g.nodes[u]
        inv = udata['pre_inv'].copy()
        for x in g.predecessors(u):
            inv = inv.join(g.nodes[x]['post_inv'])
        inv.single_block_update(udata['block'])
        if inv != udata['post_inv']:
            udata['post_inv'] = inv
            wl.update(g.successors(u))


def test():
    import code_examples
    cfg = make_tacblock_cfg(code_examples.simple, propagate_consts=True, liveness=False)
    for n in sorted(cfg.nodes()):
        block = cfg.nodes[n]['block']
        # print(cfg.nodes[n]['pre_inv'].constants)
        print_block(n, block)
        # print(cfg.nodes[n]['post_inv'].constants)


if __name__ == '__main__':
    test()
