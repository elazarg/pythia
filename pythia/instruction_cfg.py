import dis
import math
from dis import Instruction
from typing import Iterable, Any

import pythia.graph_utils as gu

Cfg = gu.Cfg[Instruction]


def is_sequencer(ins: Instruction) -> bool:
    return ins.opname in (
        'RETURN_VALUE', 'CONTINUE_LOOP', 'BREAK_LOOP', 'RAISE_VARARGS', 'JUMP_FORWARD', 'JUMP_ABSOLUTE')


def is_return(ins: Instruction) -> bool:
    return ins.opname == 'RETURN_VALUE'


def is_jump_source(ins: Instruction) -> bool:
    return is_jump(ins) or is_sequencer(ins)
    # YIELD_VALUE does not interrupt the flow
    # exceptions do not count, since they can occur practically anywhere


def is_jump(ins: Instruction) -> bool:
    return ins.opcode in dis.hasjrel or ins.opcode in dis.hasjabs


def next_list(ins: Instruction, fallthrough: gu.Label, stack_effect: int) -> list[tuple[gu.Label, int]]:
    if is_raise(ins):
        return []
    if is_for_iter(ins):
        assert fallthrough is not None
        return [(fallthrough, stack_effect),
                (ins.argval, -1)]
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
    if ins.opname in ['SETUP_EXCEPT', 'SETUP_FINALLY', 'POP_EXCEPT', 'END_FINALLY']:
        assert False, 'for all we know. we assume no exceptions'
    if is_raise(ins):
        # if we wish to analyze exception path, we should break to except: and push 3, or something.
        return -1
    if ins.opname == 'BREAK_LOOP' and ins.argrepr.startswith('FOR'):
        return -1
    return dis.stack_effect(ins.opcode, ins.arg)


def is_for_iter(ins: Instruction) -> bool:
    return ins.opname == 'FOR_ITER'


def is_raise(ins: Instruction) -> bool:
    return ins.opname == 'RAISE_VARARGS'


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
            dijkstra = gu.single_source_dijkstra_path_length(backwards_cfg, source=label, weight='stack_effect')
            res.update({k: v+1 for k, v in dijkstra.items()})
    res.update(gu.single_source_dijkstra_path_length(cfg, source=0, weight='stack_effect'))
    return res


def make_instruction_block_cfg(instructions: Iterable[Instruction]) -> tuple[dict[gu.Label, int], Cfg]:
    instructions = list(instructions)

    next_instruction: list[gu.Label] = [instructions[i + 1].offset for i in range(len(instructions) - 1)]
    next_instruction.append(math.inf)

    dbs: dict[gu.Label, Instruction] = {ins.offset: ins for ins in instructions}
    edges = [(ins.offset, dbs[j].offset, {'stack_effect': stack_effect})
             for i, ins in enumerate(instructions)
             for (j, stack_effect) in next_list(ins, fallthrough=next_instruction[i], stack_effect=calculate_stack_effect(ins))
             if dbs.get(j) is not None]
    cfg: Cfg = gu.Cfg(edges, blocks={k: [v] for k, v in dbs.items()})
    depths = calculate_stack_depth(cfg)
    # each node will hold a block of dictionaries - instruction and stack_depth
    return depths, cfg


def make_instruction_block_cfg_from_function(f: Any) -> tuple[dict[gu.Label, int], Cfg]:
    return make_instruction_block_cfg(dis.get_instructions(f))
