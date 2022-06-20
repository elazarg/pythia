# Data flow analysis and stuff.

from __future__ import annotations

import math
from typing import TypeVar

import disassemble
import graph_utils as gu
import tac
import tac_analysis_types
from tac_analysis_constant import ConstLattice, Constant

from tac_analysis_domain import IterationStrategy, VarAnalysis, BackwardIterationStrategy, ForwardIterationStrategy, \
    Analysis
from tac_analysis_liveness import LivenessLattice, Liveness
from tac_analysis_pointer import PointerAnalysis, pretty_print_pointers
from tac_analysis_types import TypeLattice


T = TypeVar('T')


def make_tacblock_cfg(f, simplify=True):
    cfg = tac.make_tacblock_cfg(f)
    if simplify:
        cfg = gu.simplify_cfg(cfg)
    return cfg


def analyze(_cfg: gu.Cfg, analysis: Analysis[T], annotations: dict[tac.Var, str]) -> None:
    name = analysis.name()
    for label in _cfg.labels:
        _cfg[label].pre[name] = analysis.bottom()
        _cfg[label].post[name] = analysis.bottom()

    cfg: IterationStrategy = BackwardIterationStrategy(_cfg) if analysis.backward else ForwardIterationStrategy(_cfg)

    wl = [entry] = {cfg.entry_label}
    cfg[entry].pre[name] = analysis.initial(analysis.top() if analysis.backward else annotations)
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


def run(f, print_analysis=False, simplify=True, module=False):
    cfg = make_tacblock_cfg(f, simplify=simplify)
    annotations = {}
    if not module:
        annotations = {tac.Var(k): v for k, v in f.__annotations__.items()}
    # analyze(cfg, LivenessDomain)
    # analyze(cfg, AliasDomain)
    # for label, block in cfg.items():
    #     rewrite_remove_useless_movs_pairs(block, label)
    #     rewrite_aliases(block, label)
    #     rewrite_remove_useless_movs(block, label)
    liveness = LivenessLattice()
    constant = ConstLattice()
    var_analysis = VarAnalysis[tac.Var, tac_analysis_types.TypeElement](TypeLattice())

    analyze(cfg, VarAnalysis[tac.Var, Liveness](liveness, backward=True), annotations)
    # analyze(cfg, ConstantDomain)
    analyze(cfg, VarAnalysis[tac.Var, Constant](constant), annotations)
    analyze(cfg, var_analysis, annotations)
    analyze(cfg, PointerAnalysis(var_analysis), annotations)
    return cfg


def print_analysis(cfg):
    for label, block in sorted(cfg.items()):
        if math.isinf(label):
            continue
        if print_analysis:
            print('Pre:')
            for k in block.pre:
                if k == 'Pointer':
                    print(f'\t{k}:', pretty_print_pointers(block.pre[k]))
                else:
                    print(f'\t{k}:', block.pre[k])
        gu.print_block(label, block)
        if print_analysis:
            print('Post:')
            for k in block.post:
                if k == 'Pointer':
                    print(f'\t{k}:', pretty_print_pointers(block.post[k]))
                else:
                    print(f'\t{k}:', block.post[k])
            print()

    if tac_analysis_types.unseen:
        print("Unseen:")
        for k, v in tac_analysis_types.unseen.items():
            print(k, v)

    # cfg.draw()


if __name__ == '__main__':
    # env, imports = disassemble.read_function('examples/feature_selection_pymm.py', 'do_work')
    # env, imports = disassemble.read_function('examples/simple.py')
    env, imports = disassemble.read_function('examples/imports.py', 'foo')

    cfg = run(imports, print_analysis=True, simplify=True, module=True)
    print_analysis(cfg)

    for k, func in env.items():
        cfg = run(func, print_analysis=True, simplify=True)
        print_analysis(cfg)
