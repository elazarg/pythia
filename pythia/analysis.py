# Data flow analysis and stuff.
import math
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


def analyze[
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


def print_analysis(
    cfg: Cfg,
    invariants: dict[str, InvariantTriple],
    loop_end: typing.Optional[Location],
    dirty_locals: set[str],
    print_invariants: bool = True,
) -> None:
    for label, block in sorted(cfg.items()):
        if math.isinf(label):
            continue
        if print_invariants:
            print("Pre:")
            for name, invariant_pair in invariants.items():
                pre_invariant = invariant_pair.pre[label]
                print(f"\t{name}: ", end="")
                print(str(pre_invariant))
        gu.print_block(label, block, cfg.annotator)
        if print_invariants:
            print("Post:")
            for name, invariant_pair in invariants.items():
                post_invariant = invariant_pair.post[label]
                print(f"\t{name}:", end="")
                print(str(post_invariant))
            print()
        if loop_end is not None and label == loop_end:
            print(f"Dirty Locals:", ", ".join(dirty_locals))
            print()
        print("Successors:", list(cfg.successors(label)))
        print()


def run(
    cfg: Cfg,
    for_location: typing.Optional[Location],
    module_type: ts.Module,
    function_name: str,
) -> dict[str, InvariantTriple]:
    liveness_invariants = analyze(cfg, LivenessVarLattice(), keep_intermediate=True)

    typed_pointer_analysis = typed_pointer.TypedPointerLattice(
        liveness_invariants.intermediate, function_name, module_type, for_location
    )
    typed_pointer_invariants = analyze(
        cfg, typed_pointer_analysis, keep_intermediate=False
    )

    invariant_pairs: dict[str, InvariantTriple] = {
        LIVENESS_INV_NAME: liveness_invariants,
        TYPE_INV_NAME: typed_pointer_invariants,
    }

    return invariant_pairs


def find_dirty_roots(
    invariants: dict[str, InvariantTriple], loop_end: typing.Optional[gu.Label]
) -> set[str]:
    if loop_end is None:
        return set()
    return set(
        typed_pointer.find_dirty_roots(
            invariants[TYPE_INV_NAME].post[loop_end],
            invariants[LIVENESS_INV_NAME].post[loop_end],
        )
    )


def analyze_function(
    filename: str,
    *function_names: str,
    print_invariants: bool,
    outfile: str,
    simplify: bool,
) -> None:
    print(filename, function_names, print_invariants, outfile, simplify)
    functions, imports = disassemble.read_file(
        filename, filter_for_loops=not bool(function_names)
    )
    module_type = ts.parse_file(filename)

    if not function_names:
        if not functions:
            raise ValueError("No functions with for loops found")
        function_names = tuple(functions.keys())

    dirty_map: dict[str, set[str]] = {}
    for function_name in function_names:
        print(function_name)
        f = functions[function_name]
        cfg = tac.make_tac_cfg(f)
        cfg = gu.simplify_cfg(cfg)
        if not simplify:
            cfg = gu.refine_to_chain(cfg)
        # gu.pretty_print_cfg(cfg)
        try:
            for_location, loop_end = gu.find_first_for_loop(
                cfg, lambda b: isinstance(b, tac.For)
            )
        except ValueError:
            for_location, loop_end = None, None

        invariant_pairs = run(
            cfg, for_location, module_type=module_type, function_name=function_name
        )

        dirty_map[function_name] = find_dirty_roots(invariant_pairs, loop_end)

        if print_invariants:
            print_analysis(cfg, invariant_pairs, loop_end, dirty_map[function_name])

    output = ast_transform.transform(filename, dirty_map)
    if outfile is None:
        print(output)
    else:
        with open(outfile, "w", encoding="utf-8") as f:
            print(output, file=f)
