# Data flow analysis and stuff.

from __future__ import annotations

import typing

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
        for label in sorted(cfg.nodes()):
            block = list(single_block_liveness(cfg[label]))
            block.reverse()
            cfg[label] = block
    return cfg


def print_block(n, block):
    print(n, ':')
    for ins in block:
        print('\t', ins)


def analyze(cfg: gu.Cfg, Analysis: typing.Type[AbstractDomain]) -> None:
    gu.set_node_attributes(cfg, {v: Analysis.bottom() for v in cfg.nodes()}, 'post_inv')
    gu.set_node_attributes(cfg, {v: Analysis.bottom() for v in cfg.nodes()}, 'pre_inv')

    cfg.entry['pre_inv'] = Analysis.top()
    wl = {0}
    while wl:
        label = wl.pop()
        node = cfg.nodes[label]

        invariant = node['pre_inv'].copy()
        for ins in cfg[label]:
            invariant.transfer(ins)
        node['post_inv'] = invariant.copy()

        for succ in cfg.successors(label):
            succ_node = cfg.nodes[succ]
            if not (invariant <= succ_node['pre_inv']):
                succ_node['pre_inv'] = succ_node['pre_inv'].join(invariant)
                wl.add(succ)


def test():
    import code_examples
    cfg = make_tacblock_cfg(code_examples.simple_loop, propagate_consts=True, liveness=False, simplify=True)
    for n in sorted(cfg.nodes()):
        block = cfg[n]
        print(cfg.nodes[n]['pre_inv'])
        print_block(n, block)
        print(cfg.nodes[n]['post_inv'])


if __name__ == '__main__':
    test()
