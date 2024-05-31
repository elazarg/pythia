# Data flow analysis and stuff.
import math
import typing
from copy import deepcopy
from dataclasses import dataclass
from typing import TypeVar, TypeAlias

import pythia.analysis_domain as domain
import pythia.graph_utils as gu
import pythia.type_system as ts
from pythia import disassemble, ast_transform
from pythia import tac
from pythia.analysis_domain import InvariantMap
from pythia.analysis_liveness import LivenessVarLattice
from pythia import analysis_typed_pointer as typed_pointer
from pythia.graph_utils import Location

Inv = TypeVar("Inv")
Cfg: TypeAlias = gu.Cfg[tac.Tac]


def make_tac_cfg(f: typing.Any, simplify: bool = True) -> Cfg:
    cfg = tac.make_tac_cfg(f)
    if simplify:
        cfg = gu.simplify_cfg(cfg)
    return cfg


@dataclass
class InvariantPair(typing.Generic[Inv]):
    pre: InvariantMap[Inv]
    post: InvariantMap[Inv]


def analyze(_cfg: Cfg, analysis: domain.InstructionLattice[Inv]) -> InvariantPair[Inv]:
    pre_result: InvariantMap[Inv] = {}
    post_result: InvariantMap[Inv] = {}

    cfg: domain.IterationStrategy = (
        domain.BackwardIterationStrategy(_cfg)
        if analysis.backward
        else domain.ForwardIterationStrategy(_cfg)
    )
    # gu.pretty_print_cfg(_cfg)
    wl = [entry] = {cfg.entry_label}
    initial = analysis.initial()
    pre_result[(entry, cfg[entry].first_index())] = initial
    while wl:
        label = min(wl)
        wl.remove(label)
        block = cfg[label]

        location = (label, block.first_index())
        pre = pre_result[location]
        post = invariant = post_result[location] = deepcopy(pre)
        for index, ins in block.items():
            location = (label, index)
            pre_result[location] = post
            try:
                invariant = analysis.transfer(invariant, ins, location)
            except Exception as e:
                e.add_note(f"At {location}")
                e.add_note(f"from {ins}")
                e.add_note(f"original command: {_cfg.annotator(location, ins)}")
                e.add_note(f"pre: {pre_result[location]}")
                raise e

            post = post_result[location] = invariant
        for next_label in cfg.successors(label):
            next_location = (next_label, cfg[next_label].first_index())
            next_pre = pre_result.get(next_location, analysis.bottom())

            # print('---')
            # print('post', post)
            # print('next_pre', next_pre)
            # print(analysis.is_less_than(post, next_pre))
            # print(analysis.join(post, next_pre))
            # print('====')
            # print()
            if not analysis.is_less_than(post, next_pre):
                # print("At label", label, "next label", next_label, "next pre", next_pre, "post", post)
                pre_result[next_location] = analysis.join(post, next_pre)
                wl.add(next_label)

    pre_result, post_result = cfg.order((pre_result, post_result))
    return InvariantPair(pre_result, post_result)


def analyze_single(
    cfg: Cfg, analysis: typing.Callable[[tac.Tac, Location], Inv]
) -> InvariantMap[Inv]:
    result: InvariantMap[Inv] = {}
    # gu.pretty_print_cfg(cfg)
    for label, block in cfg.items():
        for index, ins in block.items():
            location = (label, index)
            result[location] = analysis(ins, location)

    return result


def print_analysis(
    cfg: Cfg,
    invariants: dict[str, InvariantPair],
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
                pre_invariant = invariant_pair.pre[(label, block.first_index())]
                print(f"\t{name}:")
                pre_invariant.print()
        gu.print_block(label, block, cfg.annotator)
        if print_invariants:
            print("Post:")
            for name, invariant_pair in invariants.items():
                post_invariant = invariant_pair.post[(label, block.last_index())]
                print(f"\t{name}:")
                post_invariant.print()
            print()
        if loop_end is not None and label == loop_end[0]:
            print(f"Dirty Locals:", ", ".join(dirty_locals))
            print()
        print("Successors:", list(cfg.successors(label)))
        print()


def run(
    cfg: Cfg,
    for_location: typing.Optional[Location],
    module_type: ts.Module,
    function_name: str,
) -> dict[str, InvariantPair]:
    # gu.pretty_print_cfg(cfg)
    liveness_invariants = analyze(cfg, LivenessVarLattice())

    typed_pointer_analysis = typed_pointer.TypedPointerLattice(
        liveness_invariants.post, function_name, module_type, for_location
    )
    typed_pointer_invariants = analyze(cfg, typed_pointer_analysis)

    invariant_pairs: dict[str, InvariantPair] = {
        "Liveness": liveness_invariants,
        "TypedPointer": typed_pointer_invariants,
    }

    return invariant_pairs


def find_dirty_roots(
    invariants: dict[str, InvariantPair], loop_end: typing.Optional[Location]
) -> set[str]:
    if loop_end is None:
        return set()
    return set(
        typed_pointer.find_dirty_roots(
            invariants["TypedPointer"].post[loop_end],
            invariants["Liveness"].post[loop_end],
        )
    )


def analyze_function(
    filename: str,
    *function_names: str,
    print_invariants: bool,
    outfile: str,
    simplify: bool,
) -> None:
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
        f = functions[function_name]
        cfg = make_tac_cfg(f, simplify=False)
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
