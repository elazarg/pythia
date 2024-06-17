# Data flow analysis and stuff.
import math
import sys
import typing
from copy import deepcopy
from dataclasses import dataclass

from pythia.analysis_domain import iteration_strategy, InstructionLattice
import pythia.graph_utils as gu
import pythia.type_system as ts
from pythia import disassemble, ast_transform
from pythia import tac
from pythia.analysis_domain import InvariantMap
from pythia.analysis_liveness import LivenessVarLattice
from pythia import analysis_typed_pointer as typed_pointer
from pythia.graph_utils import Location

type Cfg = gu.Cfg[tac.Tac]

TYPE_INV_NAME = typed_pointer.TypedPointerLattice.name()
LIVENESS_INV_NAME = LivenessVarLattice.name()


@dataclass
class InvariantTriple[Inv]:
    pre: InvariantMap[Inv]
    intermediate: dict[Location, Inv]
    post: InvariantMap[Inv]


def abstract_interpretation[
    Inv
](
    _cfg: Cfg, analysis: InstructionLattice[Inv], keep_intermediate=False
) -> InvariantTriple[Inv]:
    pre_result: InvariantMap[Inv] = {}
    post_result: InvariantMap[Inv] = {}
    intermediate_result: dict[Location, Inv] = {}

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
            except Exception as e:
                e.add_note(f"At {location}")
                e.add_note(f"from {ins}")
                e.add_note(f"original command: {_cfg.annotator(location, ins)}")
                e.add_note(f"pre: {invariant}")
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
        if math.isinf(label):
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
            print(f"Dirty Locals:", ", ".join(dirty))
            print()
        print("Successors:", list(cfg.successors(label)))
        print()


def find_for_loops(cfg: Cfg) -> frozenset[tuple[gu.Label, int]]:
    return frozenset(
        {
            (label, i)
            for label, block in cfg.items()
            for i, ins in enumerate(block)
            if isinstance(ins, tac.For) and gu.find_loop_end(cfg, label) is not None
        }
    )


def run(cfg: Cfg, module_type: ts.Module, function_name: str) -> AnalysisResult:
    liveness_invariants = abstract_interpretation(
        cfg, LivenessVarLattice(), keep_intermediate=True
    )

    for_locations = find_for_loops(cfg)

    typed_pointer_analysis = typed_pointer.TypedPointerLattice(
        liveness_invariants.intermediate, function_name, module_type, for_locations
    )

    typed_pointer_invariants = abstract_interpretation(
        cfg, typed_pointer_analysis, keep_intermediate=False
    )

    dirty_map = {}
    for loop_start, i in for_locations:
        loop_end = gu.find_loop_end(cfg, loop_start)
        dirty_map[(loop_start, i)] = set(
            typed_pointer.find_dirty_roots(
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
    filename: str,
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

        analysis_result[function_name] = run(
            cfg, module_type=module_type, function_name=function_name
        )

        if print_invariants:
            print_analysis(cfg, analysis_result[function_name])

    return analysis_result


def analyze_and_transform(
    filename: str,
    function_name: typing.Optional[str],
    print_invariants: bool = False,
    outfile: typing.Optional[str] = None,
    simplify: bool = True,
) -> None:
    analysis_result = analyze_function(
        filename, function_name, print_invariants=print_invariants, simplify=simplify
    )
    output = ast_transform.transform(
        filename,
        {
            function_name: {
                result.cfg[label][i].original_lineno: dirty
                for (label, i), dirty in result.dirty_map.items()
            }
            for function_name, result in analysis_result.items()
        },
    )

    if outfile is None:
        print(output)
    else:
        with open(outfile, "w", encoding="utf-8") as f:
            print(output, file=f)
