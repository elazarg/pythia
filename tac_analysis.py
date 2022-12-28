# Data flow analysis and stuff.

from __future__ import annotations as _

import math
import typing
from dataclasses import dataclass
from typing import TypeVar, TypeAlias

import disassemble
import graph_utils as gu
import tac
import tac_analysis_domain as domain
import tac_analysis_types
from tac_analysis_constant import ConstLattice, Constant

from tac_analysis_liveness import LivenessVarLattice
from tac_analysis_pointer import PointerLattice, pretty_print_pointers, mark_reachable
from tac_analysis_types import TypeLattice
import type_system as ts


T = TypeVar('T')
Cfg: TypeAlias = gu.Cfg[tac.Tac]


def make_tac_cfg(f, simplify=True):
    cfg = tac.make_tac_cfg(f)
    if simplify:
        cfg = gu.simplify_cfg(cfg)
    return cfg


@dataclass
class InvariantPair(typing.Generic[T]):
    pre: domain.InvariantMap[T]
    post: domain.InvariantMap[T]


def analyze(_cfg: Cfg, analysis: domain.InstructionLattice[T], annotations) -> InvariantPair:
    pre_result: domain.InvariantMap = {}
    post_result: domain.InvariantMap = {}

    cfg: domain.IterationStrategy = domain.BackwardIterationStrategy(_cfg) if analysis.backward else domain.ForwardIterationStrategy(_cfg)

    wl = [entry] = {cfg.entry_label}
    initial = analysis.initial(analysis.top() if analysis.backward else annotations)
    pre_result[(entry, cfg[entry].first_index())] = initial
    while wl:
        label = wl.pop()
        block = cfg[label]

        pre = pre_result[(label, block.first_index())]
        post = invariant = pre.copy()
        for index, ins in block.items():
            location = (label, index)
            pre_result[location] = post
            invariant = analysis.transfer(invariant, ins, location)

            post = post_result[location] = invariant

        for succ in cfg.successors(label):
            next_index = (succ, cfg[succ].first_index())
            next_pre = pre_result.get(next_index, analysis.bottom())
            if not analysis.is_less_than(post, next_pre):
                pre_result[next_index] = analysis.join(post, next_pre)
                wl.add(succ)
    pre_result, post_result = cfg.order((pre_result, post_result))
    return InvariantPair(pre_result, post_result)

def run(f, functions, imports, module_type, simplify=True) -> tuple[Cfg, dict[str, InvariantPair]]:
    cfg = make_tac_cfg(f, simplify=simplify)

    # gu.pretty_print_cfg(cfg)

    annotations = {tac.Var(k): v for k, v in f.__annotations__.items()}
    # for label, block in cfg.items():
    #     rewrite_remove_useless_movs_pairs(block, label)
    #     rewrite_aliases(block, label)
    #     rewrite_remove_useless_movs(block, label)
    liveness_invariants = analyze(cfg, LivenessVarLattice(), annotations)
    constant_invariants = analyze(cfg, domain.VarLattice[Constant](ConstLattice(), liveness_invariants.post), annotations)

    type_analysis = domain.VarLattice[ts.TypeExpr](TypeLattice(f.__name__, module_type, functions, imports), liveness_invariants.post)
    type_invariants = analyze(cfg, type_analysis, annotations)

    pointer_analysis = PointerLattice(type_invariants.pre, type_analysis, liveness_invariants.post)
    pointer_invariants = analyze(cfg, pointer_analysis, annotations)

    for label, block in cfg.items():
        if not block:
            continue
        if isinstance(block[0], tac.For):
            assert len(block) == 1
            ptr = pointer_invariants.post[(label, 0)]
            alive = set(liveness_invariants.post[(label, 0)].keys())
            mark_reachable(ptr, alive, annotations, get_ins=lambda label, index: cfg[label][index])
            break

    invariants = {
        "Liveness": liveness_invariants,
        "Constant": constant_invariants,
        "Type": type_invariants,
        "Pointer": pointer_invariants,
    }

    return cfg, invariants


def print_analysis(cfg: Cfg, invariants: dict[str, InvariantPair], print_invariants: bool = True) -> None:
    for label, block in sorted(cfg.items()):
        if math.isinf(label):
            continue
        if print_invariants:
            print('Pre:')
            for name, invariant_pair in invariants.items():
                pre_invariant = invariant_pair.pre[(label, 0)]
                if name == 'Pointer':
                    print(f'\t{name}:', pretty_print_pointers(pre_invariant))
                else:
                    print(f'\t{name}:', pre_invariant)
        gu.print_block(label, block, cfg.annotator)
        if print_invariants:
            print('Post:')
            for name, invariant_pair in invariants.items():
                post_invariant = invariant_pair.post[(label, block.last_index())]
                if name == 'Pointer':
                    print(f'\t{name}:', pretty_print_pointers(post_invariant))
                else:
                    print(f'\t{name}:', post_invariant)
            print()

    if tac_analysis_types.unseen:
        print("Unseen:")
        for k, v in tac_analysis_types.unseen.items():
            print(k, v)


def analyze_function(filename: str, function_name: str) -> None:
    functions, imports = disassemble.read_file(filename)
    module_type = ts.parse_file(filename)
    cfg, invariants = run(functions[function_name],
                          functions=functions,
                          imports=imports,
                          module_type=module_type,
                          simplify=True)
    print_analysis(cfg, invariants)


def main() -> None:
    # analyze_function('examples/tests.py', 'access')
    # analyze_function('examples/tests.py', 'iterate')
    # analyze_function('examples/tests.py', 'tup')
    # analyze_function('examples/tests.py', 'destruct')
    # analyze_function('examples/feature_selection.py', 'do_work')
    analyze_function('examples/toy.py', 'minimal')
    # analyze_function('examples/toy.py', 'not_so_minimal')
    # analyze_function('examples/feature_selection.py', 'run')
    # analyze_function('examples/toy.py', 'toy3')


if __name__ == '__main__':
    main()
