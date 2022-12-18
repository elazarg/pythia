# Data flow analysis and stuff.

from __future__ import annotations as _

import math
from typing import TypeVar, TypeAlias

import disassemble
import graph_utils as gu
import tac
import tac_analysis_domain
import tac_analysis_types
from tac_analysis_constant import ConstLattice, Constant

from tac_analysis_domain import IterationStrategy, VarAnalysis, BackwardIterationStrategy, ForwardIterationStrategy, \
    Analysis
from tac_analysis_liveness import LivenessLattice, Liveness, LivenessAnalysis
from tac_analysis_pointer import PointerAnalysis, pretty_print_pointers, find_reachable
from tac_analysis_types import TypeLattice, ts

T = TypeVar('T')
Cfg: TypeAlias = gu.Cfg[tac.Tac]


def make_tac_cfg(f, simplify=True):
    cfg = tac.make_tac_cfg(f)
    if simplify:
        cfg = gu.simplify_cfg(cfg)
    return cfg


def analyze(_cfg: Cfg, analysis: Analysis[T], annotations) -> None:
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
        for index, ins in enumerate(cfg[label]):
            invariant = analysis.transfer(invariant, ins, f'{label}.{index}')

        if isinstance(invariant, tac_analysis_domain.Map):
            if analysis.name() is not LivenessLattice.name():
                liveness = block.post.get(LivenessLattice.name())
                if liveness:
                    invariant = LivenessAnalysis.remove_dead_variables(liveness, invariant)
                del liveness

        block.post[name] = invariant

        for succ in cfg.successors(label):
            next_block: gu.Block = cfg[succ]
            pre = next_block.pre[name]
            if not analysis.is_less_than(invariant, pre):
                next_block.pre[name] = analysis.join(invariant, pre)
                wl.add(succ)


def run(f, functions, imports, module_type, simplify=True) -> Cfg:
    cfg = make_tac_cfg(f, simplify=simplify)

    # gu.pretty_print_cfg(cfg)

    annotations = {tac.Var(k): v for k, v in f.__annotations__.items()}
    # for label, block in cfg.items():
    #     rewrite_remove_useless_movs_pairs(block, label)
    #     rewrite_aliases(block, label)
    #     rewrite_remove_useless_movs(block, label)
    liveness_analysis = LivenessAnalysis()
    constant_analysis = VarAnalysis[Constant](ConstLattice())
    type_analysis = VarAnalysis[ts.TypeExpr](TypeLattice(f.__name__, module_type, functions, imports))
    pointer_analysis = PointerAnalysis(type_analysis, liveness_analysis)

    analyze(cfg, liveness_analysis, annotations)
    analyze(cfg, constant_analysis, annotations)
    analyze(cfg, type_analysis, annotations)
    analyze(cfg, pointer_analysis, annotations)

    mark_heap(cfg, liveness_analysis, pointer_analysis)

    return cfg


def mark_heap(cfg: Cfg,
               liveness_analysis: VarAnalysis[Liveness],
               pointer_analysis: PointerAnalysis) -> None:
    for i, block in cfg.items():
        if not block:
            continue
        if isinstance(block[0], tac.For):
            assert len(block) == 1
            ptr = block.post[pointer_analysis.name()]
            alive = block.pre[liveness_analysis.name()]
            for var in alive.keys():
                for loc in find_reachable(ptr, var):
                    label, index = [int(x) for x in str(loc)[1:].split('.')]
                    ins = cfg[label][index]
                    ins.expr.allocation = tac.AllocationType.HEAP
            break


def print_analysis(cfg: Cfg) -> None:
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
        gu.print_block(label, block, cfg.annotator)
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


def analyze_function(filename: str, function_name: str) -> None:
    functions, imports = disassemble.read_file(filename)
    module_type = ts.parse_file(filename)
    cfg = run(functions[function_name],
              functions=functions,
              imports=imports,
              module_type=module_type,
              simplify=True)
    print_analysis(cfg)


def main() -> None:
    # analyze_function('examples/tests.py', 'listing')
    # analyze_function('examples/feature_selection.py', 'do_work')
    # analyze_function('examples/feature_selection.py', 'run')
    # analyze_function('examples/toy.py', 'destruct')
    analyze_function('examples/toy.py', 'minimal')
    # analyze_function('examples/toy.py', 'not_so_minimal')
    # analyze_function('examples/toy.py', 'toy3')


if __name__ == '__main__':
    main()
