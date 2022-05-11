# Data flow analysis and stuff.

from __future__ import annotations

import math
import typing

import graph_utils as gu
import tac
import tac_analysis_types

from tac_analysis_domain import AbstractDomain, IterationStrategy, Lattice
from tac_analysis_liveness import LivenessDomain, rewrite_remove_useless_movs, rewrite_remove_useless_movs_pairs
from tac_analysis_constant import ConstantDomain
from tac_analysis_pointer import PointerDomain
from tac_analysis_alias import AliasDomain, rewrite_aliases
from tac_analysis_types import TypeDomain


def make_tacblock_cfg(f, simplify=True):
    cfg = tac.make_tacblock_cfg(f)
    if simplify:
        cfg = gu.simplify_cfg(cfg)
    return cfg


def print_block(n, block):
    print(n, ':')
    for ins in block:
        print('\t', ins)


def analyze(_cfg: gu.Cfg, Analysis: typing.Type[AbstractDomain], initial: Lattice = None) -> None:
    name = Analysis.name()
    for label in _cfg.labels:
        _cfg[label].pre[name] = Analysis.bottom()
        _cfg[label].post[name] = Analysis.bottom()

    cfg: IterationStrategy = Analysis.view(_cfg)

    wl = [entry] = {cfg.entry_label}
    cfg[entry].pre[name] = initial or Analysis.initial()
    while wl:
        label = wl.pop()
        block = cfg[label]

        invariant = block.pre[name].copy()
        for index, tac in enumerate(cfg[label]):
            invariant.transfer(tac, f'{label}.{index}')

        if Analysis is not LivenessDomain:
            liveness = typing.cast(typing.Optional[LivenessDomain], block.post.get(LivenessDomain.name()))
            if liveness:
                invariant.keep_only_live_vars(liveness.vars)
            del liveness

        block.post[name] = invariant

        for succ in cfg.successors(label):
            next_block: gu.Block = cfg[succ]
            if not (invariant <= (pre := next_block.pre[name])):
                next_block.pre[name] = pre.join(invariant)
                wl.add(succ)


def test(f: type(test), print_analysis=False, simplify=True):
    cfg = make_tacblock_cfg(f, simplify=simplify)

    analyze(cfg, LivenessDomain)
    analyze(cfg, AliasDomain)
    for label, block in cfg.items():
        rewrite_aliases(block, label)
        rewrite_remove_useless_movs_pairs(block, label)
        rewrite_remove_useless_movs(block, label)
    analyze(cfg, LivenessDomain)
    analyze(cfg, ConstantDomain)
    analyze(cfg, TypeDomain, TypeDomain.read_initial(f.__annotations__))
    analyze(cfg, PointerDomain)

    for label, block in sorted(cfg.items()):
        if math.isinf(label):
            continue
        if print_analysis:
            print('Pre:')
            print('\t', block.pre[LivenessDomain.name()])
            print('\t', block.pre[PointerDomain.name()])
            print('\t', block.pre[ConstantDomain.name()])
            print('\t', block.pre[TypeDomain.name()])
        print_block(label, block)
        if print_analysis:
            print('Post:')
            print('\t', block.post[LivenessDomain.name()])
            print('\t', block.post[PointerDomain.name()])
            print('\t', block.post[ConstantDomain.name()])
            print('\t', block.post[TypeDomain.name()])
            print()

    print("Unseen:")
    for k, v in tac_analysis_types.unseen.items():
        print(k, v)


if __name__ == '__main__':
    import code_examples
    test(code_examples.feature_selection, print_analysis=True, simplify=True)
