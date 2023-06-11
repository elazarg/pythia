# Data flow analysis and stuff.

from __future__ import annotations as _

import math
import typing
from dataclasses import dataclass
from typing import TypeVar, TypeAlias

import pythia.analysis_domain as domain
import pythia.graph_utils as gu
import pythia.type_system as ts
from pythia import disassemble, ast_transform
from pythia import tac
from pythia.analysis_allocation import AllocationType, AllocationChecker
from pythia.analysis_constant import ConstLattice
from pythia.analysis_dirty import DirtyLattice, find_reaching_locals
from pythia.analysis_domain import InvariantMap
from pythia.analysis_liveness import LivenessVarLattice
from pythia.analysis_pointer import PointerLattice, pretty_print_pointers, update_allocation_invariants
from pythia.analysis_types import TypeLattice
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
            # assert not isinstance(invariant, domain.Bottom), f'At {location}\nfrom {pre_result[location]}\n{ins} produced bottom'

            post = post_result[location] = invariant
        for next_label in cfg.successors(label):
            next_location = (next_label, cfg[next_label].first_index())
            next_pre = pre_result.get(next_location, analysis.bottom())
            if not analysis.is_less_than(post, next_pre):
                # print("At label", label, "next label", next_label, "next pre", next_pre, "post", post)
                pre_result[next_location] = analysis.join(post, next_pre)
                wl.add(next_label)

    pre_result, post_result = cfg.order((pre_result, post_result))
    return InvariantPair(pre_result, post_result)


def analyze_single(cfg: Cfg, analysis: typing.Callable[[tac.Tac, Location], T]) -> InvariantMap[T]:
    result: InvariantMap[T] = {}
    gu.pretty_print_cfg(cfg)
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
        print("Successors:", list(cfg.successors(label)))
        print()


def find_first_for_loop(cfg: Cfg) -> tuple[Location, Location]:
    first_label = min(label for label, block in cfg.items()
                      if block and isinstance(block[0], tac.For))
    block = cfg[first_label]
    assert len(block) == 1
    prev, after = cfg.predecessors(first_label)
    return ((first_label, 0), (after, cfg[after].last_index()))


def run(cfg: Cfg, annotations: dict[tac.Var, str], module_type: ts.Module, function_name: str) -> tuple[InvariantMap[AllocationType], set[str], dict[str, InvariantPair]]:

    liveness_invariants = analyze(cfg, LivenessVarLattice(), annotations)

    type_analysis: domain.VarLattice[ts.TypeExpr] = domain.VarLattice[ts.TypeExpr](TypeLattice(function_name, module_type),
                                                                                   liveness_invariants.post)
    type_invariants = analyze(cfg, type_analysis, annotations)
    allocation_invariants: InvariantMap[AllocationType] = analyze_single(cfg, AllocationChecker(type_invariants.pre, type_analysis))

    pointer_analysis = PointerLattice(allocation_invariants, liveness_invariants.post)
    pointer_invariants = analyze(cfg, pointer_analysis, annotations)

    dirty_analysis = DirtyLattice(pointer_invariants.post, allocation_invariants)
    dirty_invariants = analyze(cfg, dirty_analysis, annotations)

    for_location, loop_end = find_first_for_loop(cfg)

    update_allocation_invariants(allocation_invariants,
                                 pointer_invariants.post[for_location],
                                 liveness_invariants.post[for_location],
                                 annotations)

    dirty_locals = set(find_reaching_locals(pointer_invariants.post[loop_end],
                                            liveness_invariants.post[loop_end],
                                            dirty_invariants.post[loop_end]))

    invariant_pairs: dict[str, InvariantPair] = {
        "Liveness": liveness_invariants,
        "Type": type_invariants,
        "Pointer": pointer_invariants,
        "Dirty": dirty_invariants,
    }

    return allocation_invariants, dirty_locals, invariant_pairs


def analyze_function(filename: str, *function_names: str, print_invariants: bool, outfile: str, simplify: bool) -> None:
    functions, imports = disassemble.read_file(filename)
    module_type = ts.parse_file(filename)

    dirty_map = {}
    for function_name in function_names:
        f = functions[function_name]
        cfg = make_tac_cfg(f, simplify=False)
        cfg = gu.simplify_cfg(cfg)
        if not simplify:
            cfg = gu.refine_to_chain(cfg)
        annotations = {tac.Var(k): v for k, v in f.__annotations__.items()}
        allocation_invariants, dirty_locals, invariant_pairs = run(cfg, annotations,
                                                                   module_type=module_type,
                                                                   function_name=function_name)

        if print_invariants:
            print_analysis(cfg, invariant_pairs, allocation_invariants)

        dirty_map[function_name] = dirty_locals

    output = ast_transform.transform(filename, dirty_map)
    if outfile is None:
        print(output)
    else:
        with open(outfile, 'w', encoding='utf-8') as f:
            print(output, file=f)
