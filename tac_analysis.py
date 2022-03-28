# Data flow analysis and stuff.

from __future__ import annotations

import typing

import graph_utils as gu
import tac

from tac_analysis_domain import AbstractDomain, IterationStrategy
from tac_analysis_liveness import LivenessDomain
from tac_analysis_constant import ConstantDomain


def make_tacblock_cfg(f, propagate_consts=True, liveness=True, propagate_assignments=True, simplify=True):
    cfg = tac.make_tacblock_cfg(f)
    if simplify:
        cfg = gu.simplify_cfg(cfg)
    if propagate_consts:
        analyze(cfg, ConstantDomain)
    if liveness:
        analyze(cfg, LivenessDomain)
    return cfg


def print_block(n, block):
    print(n, ':')
    for ins in block:
        print('\t', ins)


def analyze(_cfg: gu.Cfg, Analysis: typing.Type[AbstractDomain]) -> None:
    name = Analysis.name()
    for label in _cfg.labels:
        _cfg[label].pre[name] = Analysis.bottom()
        _cfg[label].post[name] = Analysis.bottom()

    cfg: IterationStrategy = Analysis.view(_cfg)

    wl = [entry] = {cfg.entry_label}
    cfg[entry].pre[name] = Analysis.top()
    while wl:
        label = wl.pop()
        block = cfg[label]

        invariant = block.pre[name].copy()
        for ins in cfg[label]:
            invariant.transfer(ins)
        block.post[name] = invariant.copy()

        for succ in cfg.successors(label):
            next_block: gu.Block = cfg[succ]
            if not (invariant <= (pre := next_block.pre[name])):
                next_block.pre[name] = pre.join(invariant)
                wl.add(succ)


def test():
    import code_examples
    cfg = make_tacblock_cfg(code_examples.simple_loop, propagate_consts=False, liveness=True, simplify=False)
    for label, block in sorted(cfg.items()):
        print('pre', block.pre)
        print_block(label, block)
        print('post', block.post)


if __name__ == '__main__':
    test()
