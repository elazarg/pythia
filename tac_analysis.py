# Data flow analysis and stuff.

from __future__ import annotations

import typing

import graph_utils as gu
import tac

from tac_analysis_domain import AbstractDomain, IterationStrategy
from tac_analysis_liveness import LivenessDomain
from tac_analysis_constant import ConstantDomain
from tac_analysis_pointer import PointerDomain


def make_tacblock_cfg(f, analyses: typing.Iterable[AbstractDomain], simplify=True):
    cfg = tac.make_tacblock_cfg(f)
    if simplify:
        cfg = gu.simplify_cfg(cfg)
    for analysis in analyses:
        analyze(cfg, analysis)
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
    cfg[entry].pre[name] = Analysis.initial()
    while wl:
        label = wl.pop()
        block = cfg[label]

        invariant = block.pre[name].copy()
        for ins in cfg[label]:
            invariant.transfer(ins)

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


def test(f, print_analysis=False):
    cfg = make_tacblock_cfg(f, [LivenessDomain, PointerDomain], simplify=True)
    for label, block in sorted(cfg.items()):
        if print_analysis:
            print('pre', block.pre)
        print_block(label, block)
        if print_analysis:
            print('post', block.post)


if __name__ == '__main__':
    import code_examples
    test(code_examples.gradient_descent, print_analysis=True)
