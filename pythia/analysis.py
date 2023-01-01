# Data flow analysis and stuff.

from __future__ import annotations as _

import math
import typing
from dataclasses import dataclass
from typing import TypeVar, TypeAlias

import pythia.analysis_domain as domain
import pythia.graph_utils as gu
import pythia.type_system as ts
from pythia import disassemble
from pythia import tac
from pythia.analysis_constant import ConstLattice, Constant
from pythia.analysis_dirty import DirtyLattice, Dirty
from pythia.analysis_domain import InvariantMap, MapDomain
from pythia.analysis_liveness import LivenessVarLattice, Liveness
from pythia.analysis_pointer import PointerLattice, pretty_print_pointers, find_reachable, Graph
from pythia.analysis_types import TypeLattice, AllocationChecker, AllocationType
from pythia.graph_utils import Location

T = TypeVar('T')
Cfg: TypeAlias = gu.Cfg[tac.Tac]


def make_tac_cfg(f: typing.Any, simplify: bool = True) -> Cfg:
    cfg = tac.make_tac_cfg(f)
    if simplify:
        cfg = gu.simplify_cfg(cfg)
    return cfg


@dataclass
class InvariantPair(typing.Generic[T]):
    pre: InvariantMap[T]
    post: InvariantMap[T]


def analyze(_cfg: Cfg, analysis: domain.InstructionLattice[T], annotations: dict[tac.Var, str]) -> InvariantPair[T]:
    pre_result: InvariantMap[T] = {}
    post_result: InvariantMap[T] = {}

    cfg: domain.IterationStrategy = domain.BackwardIterationStrategy(_cfg) if analysis.backward else domain.ForwardIterationStrategy(_cfg)

    wl = [entry] = {cfg.entry_label}
    initial = analysis.initial(annotations)
    pre_result[(entry, cfg[entry].first_index())] = initial
    while wl:
        label = wl.pop()
        block = cfg[label]

        location = (label, block.first_index())
        pre = pre_result[location]
        post = invariant = post_result[location] = analysis.copy(pre)
        for index, ins in block.items():
            location = (label, index)
            pre_result[location] = post
            invariant = analysis.transfer(invariant, ins, location)

            post = post_result[location] = invariant

        for next_label in cfg.successors(label):
            next_location = (next_label, cfg[next_label].first_index())
            next_pre = pre_result.get(next_location, analysis.bottom())
            if not analysis.is_less_than(post, next_pre):
                pre_result[next_location] = analysis.join(post, next_pre)
                wl.add(next_label)

    pre_result, post_result = cfg.order((pre_result, post_result))
    return InvariantPair(pre_result, post_result)


def analyze_single(cfg: Cfg, analysis: typing.Callable[[tac.Tac, Location], T]) -> InvariantMap[T]:
    result: InvariantMap[T] = {}

    for label, block in cfg.items():
        for index, ins in block.items():
            location = (label, index)
            result[location] = analysis(ins, location)

    return result


def print_analysis(cfg: Cfg, invariants: dict[str, InvariantPair], property_map: InvariantMap[AllocationType], print_invariants: bool = True) -> None:
    for label, block in sorted(cfg.items()):
        if math.isinf(label):
            continue
        if print_invariants:
            print('Pre:')
            for name, invariant_pair in invariants.items():
                pre_invariant = invariant_pair.pre[(label, block.first_index())]
                if name == 'Pointer':
                    print(f'\t{name}:', pretty_print_pointers(pre_invariant))
                else:
                    print(f'\t{name}:', pre_invariant)
        gu.print_block(label, block, cfg.annotator, lambda location, ins: property_map[location])
        if print_invariants:
            print('Post:')
            for name, invariant_pair in invariants.items():
                post_invariant = invariant_pair.post[(label, block.last_index())]
                if name == 'Pointer':
                    print(f'\t{name}:', pretty_print_pointers(post_invariant))
                else:
                    print(f'\t{name}:', post_invariant)
            print()


def run(f: typing.Any, module_type: ts.Module, simplify: bool = True) -> None:
    cfg: Cfg = make_tac_cfg(f, simplify=simplify)

    annotations = {tac.Var(k): v for k, v in f.__annotations__.items()}

    liveness_invariants: InvariantPair[MapDomain[Liveness]] = analyze(cfg, LivenessVarLattice(), annotations)
    constant_invariants: InvariantPair[MapDomain[Constant]] = analyze(cfg, domain.VarLattice(ConstLattice(), liveness_invariants.post), annotations)

    type_analysis: domain.VarLattice[ts.TypeExpr] = domain.VarLattice[ts.TypeExpr](TypeLattice(f.__name__, module_type),
                                                                                   liveness_invariants.post)
    type_invariants: InvariantPair[MapDomain[ts.TypeExpr]] = analyze(cfg, type_analysis, annotations)
    allocation_invariants: InvariantMap[AllocationType] = analyze_single(cfg, AllocationChecker(type_invariants.pre, type_analysis))

    pointer_analysis = PointerLattice(allocation_invariants, liveness_invariants.post)
    pointer_invariants: InvariantPair[Graph] = analyze(cfg, pointer_analysis, annotations)

    dirty_analysis = DirtyLattice(pointer_invariants.pre)
    dirty_invariants: InvariantPair[Dirty] = analyze(cfg, dirty_analysis, annotations)

    for label, block in cfg.items():
        if not block:
            continue
        if isinstance(block[0], tac.For):
            assert len(block) == 1
            ptr = pointer_invariants.post[(label, 0)]
            liveness_post = liveness_invariants.post[(label, 0)]
            assert not isinstance(liveness_post, domain.Bottom)
            alive = set(liveness_post.keys())
            for location in find_reachable(ptr, alive, annotations):
                allocation_invariants[location] = AllocationType.HEAP
            break

    invariant_pairs: dict[str, InvariantPair] = {
        "Liveness": liveness_invariants,
        "Constant": constant_invariants,
        "Type": type_invariants,
        "Pointer": pointer_invariants,
        "Dirty": dirty_invariants,
    }

    print_analysis(cfg, invariant_pairs, allocation_invariants)


def analyze_function(filename: str, function_name: str) -> None:
    functions, imports = disassemble.read_file(filename)
    module_type = ts.parse_file(filename)
    run(functions[function_name], module_type=module_type,
        simplify=False)



def main() -> None:
    example_dir = '../examples'
    # analyze_function(f'{example_dir}/tests.py', 'negative')
    # analyze_function(f'{example_dir}/tests.py', 'iterate')
    # analyze_function(f'{example_dir}/tests.py', 'tup')
    # analyze_function(f'{example_dir}/tests.py', 'destruct')
    analyze_function(f'{example_dir}/feature_selection.py', 'do_work')
    # analyze_function(f'{example_dir}/toy.py', 'minimal')
    # analyze_function(f'{example_dir}/toy.py', 'not_so_minimal')
    # analyze_function(f'{example_dir}/feature_selection.py', 'run')


if __name__ == '__main__':
    main()
