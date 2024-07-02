import dis
import math
from dis import Instruction
from typing import Any

import pythia.graph_utils as gu

Cfg = gu.Cfg[Instruction]


def is_unconditional_jump(ins: Instruction) -> bool:
    return ins.opname in {
        "CONTINUE_LOOP",
        "BREAK_LOOP",
        "JUMP_BACKWARD",
        "JUMP_FORWARD",
        "JUMP_ABSOLUTE",
    }


def is_return(ins: Instruction) -> bool:
    return ins.opname.startswith("RETURN_")


def is_raise(ins: Instruction) -> bool:
    return ins.opname in {"RERAISE", "RAISE_VARARGS"}


def is_sequencer(ins: Instruction) -> bool:
    return is_return(ins) or is_unconditional_jump(ins) or is_raise(ins)


def is_jump_source(ins: Instruction) -> bool:
    # YIELD_VALUE does not interrupt the flow
    # exceptions are handled through the exception table
    return is_jump(ins) or is_sequencer(ins)


def is_jump(ins: Instruction) -> bool:
    return ins.opcode in dis.hasjrel or ins.opcode in dis.hasjabs


# returns a list of (label, stack_effect) pairs
def next_list(
    ins: Instruction, fallthrough: gu.Label, stack_effect: int
) -> list[tuple[gu.Label, int]]:
    if is_raise(ins):
        return []
    if is_for_iter(ins):
        assert fallthrough is not None
        return [(fallthrough, stack_effect), (ins.argval, -1)]
    res: list[tuple[gu.Label, int]] = []
    if not is_sequencer(ins):
        assert fallthrough is not None
        res.append((fallthrough, stack_effect))
    if is_jump_source(ins):
        res.append((ins.argval, stack_effect))
    return res


def calculate_stack_effect(ins: Instruction) -> int:
    """not exact.
    see https://github.com/python/cpython/blob/master/Python/compile.c#L860"""
    assert ins.opname not in [
        "SETUP_EXCEPT",
        "SETUP_FINALLY",
        "POP_EXCEPT",
        "END_FINALLY",
    ], "for all we know. we assume no exceptions"
    if ins.opname == "END_FOR":
        return 0
    arg = ins.arg if ins.opcode in dis.hasarg else None
    res = dis.stack_effect(ins.opcode, arg)
    return res


def is_for_iter(ins: Instruction) -> bool:
    return ins.opname == "FOR_ITER"


def calculate_stack_depth(cfg: Cfg) -> dict[gu.Label, int]:
    """The stack depth is supposed to be independent of path, so dijkstra on the undirected graph suffices
    (and may be too strong, since we don't need minimality).
    We do it bidirectionally because we want to work with unreachable code too.
    """
    res: dict[gu.Label, int] = {}
    backwards_cfg = cfg.reverse(copy=False)
    for label in cfg.nodes:
        if cfg[label] and is_return(cfg[label][-1]):
            # add 1 since return also pops
            dijkstra = gu.single_source_dijkstra_path_length(
                backwards_cfg, source=label, weight="stack_effect"
            )
            res.update({k: v + 1 for k, v in dijkstra.items()})
    res.update(
        gu.single_source_dijkstra_path_length(cfg, source=0, weight="stack_effect")
    )
    return res


def pos_str(p: dis.Positions) -> str:
    return f"{p.lineno}:{p.col_offset}-{p.end_lineno}:{p.end_col_offset}"


def make_instruction_block_cfg(f: Any) -> tuple[dict[gu.Label, int], Cfg]:
    b = dis.Bytecode(f, show_caches=False)
    instructions = list(b)
    # print(b.dis())
    next_instruction: list[gu.Label] = [
        instructions[i + 1].offset for i in range(len(instructions) - 1)
    ]
    next_instruction.append(gu.EXIT_LABEL)

    dbs: dict[gu.Label, Instruction] = {ins.offset: ins for ins in instructions}
    edges = [
        (ins.offset, dbs[j].offset, {"stack_effect": stack_effect})
        for i, ins in enumerate(instructions)
        for (j, stack_effect) in next_list(
            ins,
            fallthrough=next_instruction[i],
            stack_effect=calculate_stack_effect(ins),
        )
        if dbs.get(j) is not None
    ]
    # add each exception target from b.exception_table as an edge from entry:
    for ex_entry in b.exception_entries:
        edge = (0, ex_entry.target, {"stack_effect": ex_entry.depth})
        edges.append(edge)

    cfg: Cfg = gu.Cfg(
        edges,
        blocks={k: [v] for k, v in dbs.items()},
        add_sink=True,
        add_source=False,
    )
    cfg.annotator = lambda i, ins: f"{pos_str(ins.positions)}"
    depths = calculate_stack_depth(cfg)
    cfg = gu.simplify_cfg(
        cfg, exception_labels={ex.target for ex in b.exception_entries}
    )
    # gu.pretty_print_cfg(cfg)
    # each node will hold a block of dictionaries - instruction and stack_depth
    return depths, cfg
