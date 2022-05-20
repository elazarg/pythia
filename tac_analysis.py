# Data flow analysis and stuff.

from __future__ import annotations

import math

import disassemble
import graph_utils as gu
import tac
import tac_analysis_types
from tac_analysis_constant import ConstLattice

from tac_analysis_domain import IterationStrategy, Lattice, AbstractAnalysis, Cartesian
# from tac_analysis_liveness import LivenessDomain, rewrite_remove_useless_movs, rewrite_remove_useless_movs_pairs
# from tac_analysis_constant import ConstantDomain
# from tac_analysis_pointer import PointerDomain
# from tac_analysis_alias import AliasDomain, rewrite_aliases
from tac_analysis_liveness import LivenessLattice
from tac_analysis_types import TypeLattice


def make_tacblock_cfg(f, simplify=True):
    cfg = tac.make_tacblock_cfg(f)
    if simplify:
        cfg = gu.simplify_cfg(cfg)
    return cfg


def print_block(n, block):
    print(n, ':')
    for i, ins in enumerate(block):
        label = f'{n}.{i}'
        print(f'\t{label:6}\t', ins)


def analyze(_cfg: gu.Cfg, analysis: Cartesian, annotations: dict[tac.Var, str]) -> None:
    name = analysis.name()
    for label in _cfg.labels:
        _cfg[label].pre[name] = analysis.bottom()
        _cfg[label].post[name] = analysis.bottom()

    cfg: IterationStrategy = analysis.view(_cfg)

    wl = [entry] = {cfg.entry_label}
    cfg[entry].pre[name] = analysis.initial(annotations)
    while wl:
        label = wl.pop()
        block = cfg[label]

        invariant = block.pre[name].copy()
        for index, tac in enumerate(cfg[label]):
            invariant = analysis.transfer(invariant, tac, f'{label}.{index}')

        # if analysis.name() is not "Alive":
        #     liveness = typing.cast(typing.Optional[LivenessDomain], block.post.get(LivenessDomain.name()))
        #     if liveness:
        #         invariant.keep_only_live_vars(liveness.vars)
        #     del liveness

        block.post[name] = invariant

        for succ in cfg.successors(label):
            next_block: gu.Block = cfg[succ]
            pre = next_block.pre[name]
            if not analysis.is_less_than(invariant, pre):
                next_block.pre[name] = analysis.join(invariant, pre)
                wl.add(succ)


def test(f: type(test), print_analysis=False, simplify=True):
    cfg = make_tacblock_cfg(f, simplify=simplify)
    annotations = {tac.Var(k): v for k, v in f.__annotations__.items()}
    # analyze(cfg, LivenessDomain)
    # analyze(cfg, AliasDomain)
    # for label, block in cfg.items():
    #     rewrite_remove_useless_movs_pairs(block, label)
    #     rewrite_aliases(block, label)
    #     rewrite_remove_useless_movs(block, label)
    analyze(cfg, Cartesian(LivenessLattice(), backward=True), annotations)
    # analyze(cfg, ConstantDomain)
    analyze(cfg, Cartesian(ConstLattice()), annotations)
    analyze(cfg, Cartesian(TypeLattice()), annotations)
    # analyze(cfg, PointerDomain)

    for label, block in sorted(cfg.items()):
        if math.isinf(label):
            continue
        if print_analysis:
            print('Pre:')
            for k in block.pre:
                print(f'\t{k}:', block.pre[k])
        print_block(label, block)
        if print_analysis:
            print('Post:')
            for k in block.post:
                print(f'\t{k}:', block.post[k])
            print()

    if tac_analysis_types.unseen:
        print("Unseen:")
        for k, v in tac_analysis_types.unseen.items():
            print(k, v)

    # cfg.draw()


if __name__ == '__main__':
    import code_examples
    # import dis
    # print(dis.dis(code_examples.jumps))
    code = disassemble.read_function('code_examples.py', 'feature_selection')
    test(code, print_analysis=True, simplify=False)
