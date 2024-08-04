# Data flow analysis and stuff.
import pathlib
import typing
from copy import deepcopy
from dataclasses import dataclass

from pythia.strategy import iteration_strategy
from pythia.domains import InstructionLattice
from pythia import graph_utils as gu
from pythia import type_system as ts
from pythia import tac
from pythia import disassemble, ast_transform
from pythia.domains import InvariantMap
from pythia.dom_typed_pointer import TypedPointerLattice, find_dirty_roots
from pythia.dom_liveness import LivenessVarLattice

type Cfg = gu.Cfg[tac.Tac]

TYPE_INV_NAME = TypedPointerLattice.name()
LIVENESS_INV_NAME = LivenessVarLattice.name()


@dataclass
class InvariantTriple[Inv]:
    pre: InvariantMap[Inv]
    intermediate: dict[gu.Location, Inv]
    post: InvariantMap[Inv]


def abstract_interpretation[
    Inv
](
    _cfg: Cfg, analysis: InstructionLattice[Inv], keep_intermediate=False
) -> InvariantTriple[Inv]:
    pre_result: InvariantMap[Inv] = {}
    post_result: InvariantMap[Inv] = {}
    intermediate_result: dict[gu.Location, Inv] = {}

    cfg = iteration_strategy(_cfg, analysis.backward)
    wl = [entry] = {cfg.entry_label}
    pre_result[entry] = analysis.initial()
    while wl:
        label = min(wl)
        wl.remove(label)
        block = cfg[label]

        invariant = deepcopy(pre_result[label])
        for i, (index, ins) in enumerate(block.items()):
            location = (label, index)
            if keep_intermediate:
                intermediate_result[location] = deepcopy(invariant)
            try:
                invariant = analysis.transfer(invariant, ins, location)
            except Exception as e:  # pragma: no cover
                e.add_note(f"At {location}")
                e.add_note(f"From {ins}")
                e.add_note(f"Original command: {_cfg.annotator(location, ins)}")
                e.add_note(f"Pre invariant: {invariant}")
                raise e
        post_result[label] = invariant

        for next_label in cfg.successors(label):
            next_pre_invariants = [
                post_result.get(n, analysis.bottom())
                for n in cfg.predecessors(next_label)
            ]
            next_pre = pre_result.get(next_label, analysis.bottom())
            new_next_pre = analysis.join_all(invariant, *next_pre_invariants)
            if not analysis.is_less_than(new_next_pre, next_pre):
                pre_result[next_label] = new_next_pre
                wl.add(next_label)

    pre_result, post_result = cfg.order((pre_result, post_result))
    return InvariantTriple(pre_result, intermediate_result, post_result)


@dataclass
class AnalysisResult:
    cfg: Cfg
    invariants: dict[str, InvariantTriple]
    dirty_map: dict[gu.Location, set[str]]


def print_analysis(
    cfg: Cfg,
    analysis_result: AnalysisResult,
    print_invariants: bool = True,
) -> None:
    dirty_map = {
        gu.find_loop_end(cfg, label): m
        for (label, i), m in analysis_result.dirty_map.items()
    }
    for label, block in sorted(cfg.items()):
        if gu.is_exit(label):
            continue
        if print_invariants:
            print("Pre:")
            for name, invariant_pair in analysis_result.invariants.items():
                pre_invariant = invariant_pair.pre[label]
                print(f"\t{name}: ", end="")
                print(str(pre_invariant))
        gu.print_block(label, block, cfg.annotator)
        if print_invariants:
            print("Post:")
            for name, invariant_pair in analysis_result.invariants.items():
                post_invariant = invariant_pair.post[label]
                print(f"\t{name}:", end="")
                print(str(post_invariant))
            print()
        if (dirty := dirty_map.get(label)) is not None:
            print("Dirty Locals:", ", ".join(dirty))
            print()
        print("Successors:", list(cfg.successors(label)))
        print()


def run(
    cfg: Cfg,
    module_type: ts.Module,
    function_name: str,
    for_locations: frozenset[gu.Location],
) -> AnalysisResult:
    liveness_invariants = abstract_interpretation(
        cfg, LivenessVarLattice(), keep_intermediate=True
    )

    typed_pointer_analysis = TypedPointerLattice(
        liveness_invariants.intermediate, function_name, module_type, for_locations
    )

    typed_pointer_invariants = abstract_interpretation(
        cfg, typed_pointer_analysis, keep_intermediate=False
    )

    dirty_map = {}
    for loop_start, i in for_locations:
        loop_end = gu.find_loop_end(cfg, loop_start)
        dirty_map[(loop_start, i)] = set(
            find_dirty_roots(
                typed_pointer_invariants.post[loop_end],
                liveness_invariants.post[loop_end],
            )
        )

    invariants: dict[str, InvariantTriple] = {
        LIVENESS_INV_NAME: liveness_invariants,
        TYPE_INV_NAME: typed_pointer_invariants,
    }
    return AnalysisResult(cfg, invariants, dirty_map)


def analyze_function(
    filename: pathlib.Path,
    function_name: typing.Optional[str],
    print_invariants: bool = False,
    simplify: bool = True,
) -> dict[str, AnalysisResult]:
    parsed_file = disassemble.read_file(filename)
    module_type = ts.parse_file(filename)

    functions = parsed_file.functions
    if function_name is not None:
        functions = {function_name: functions[function_name]}

    analysis_result: dict[str, AnalysisResult] = {}
    for function_name, f in functions.items():
        cfg = tac.make_tac_cfg(f, simplify=simplify)
        for_annotations = parsed_file.annotated_for[function_name]
        for_locations = gu.find_loops(
            cfg,
            is_loop=lambda ins: (
                isinstance(ins, tac.For) and ins.original_lineno in for_annotations
            ),
        )
        analysis_result[function_name] = run(
            cfg,
            module_type=module_type,
            function_name=function_name,
            for_locations=for_locations,
        )

        if print_invariants:
            print_analysis(cfg, analysis_result[function_name])

    return analysis_result


def analyze_and_transform(
    filename: pathlib.Path,
    function_name: typing.Optional[str],
    print_invariants: bool = False,
    simplify: bool = True,
) -> str:
    analysis_result = analyze_function(
        filename, function_name, print_invariants=print_invariants, simplify=simplify
    )
    dirty_map = {
        function_name: {
            result.cfg[label][i].original_lineno: dirty
            for (label, i), dirty in result.dirty_map.items()
        }
        for function_name, result in analysis_result.items()
    }
    return ast_transform.transform(filename, dirty_map)
