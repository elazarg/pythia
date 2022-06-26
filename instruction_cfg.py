from typing import Iterable

import dis
from dis import Instruction

import graph_utils as gu

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


def is_jump(ins: Instruction):
    return ins.opcode in dis.hasjrel or ins.opcode in dis.hasjabs


# static int
# instr_size(struct instr *instruction)
# {
#     int opcode = instruction->i_opcode;
#     int oparg = HAS_ARG(opcode) ? instruction->i_oparg : 0;
#     int extended_args = (0xFFFFFF < oparg) + (0xFFFF < oparg) + (0xFF < oparg);
#     int caches = _PyOpcode_Caches[opcode];
#     return extended_args + 1 + caches;
# }
def size(ins: Instruction) -> int:
    return 2
    has_arg = ins.opcode >= dis.HAVE_ARGUMENT
    oparg = self.arg if has_arg else 0
    extended_args = (0xFFFFFF < oparg) + (0xFFFF < oparg) + (0xFF < oparg)
    return extended_args + 1
    return 1 if ins.opcode < dis.HAVE_ARGUMENT else 2


def fallthrough_offset(ins: Instruction) -> int:
    return ins.offset + size(ins)


def next_list(ins: Instruction) -> list[tuple[int, int]]:
    if is_raise(ins):
        return []
    if is_for_iter(ins):
        return [(fallthrough_offset(ins), stack_effect(ins)),
                (ins.argval, -1)]
    res = []
    if not is_sequencer(ins):
        res.append((fallthrough_offset(ins), stack_effect(ins)))
    if is_jump_source(ins):
        res.append((ins.argval, stack_effect(ins)))
    return res


def is_block_boundary(ins: Instruction) -> bool:
    return is_jump_source(ins) or ins.is_jump_target


def stack_effect(ins: Instruction) -> int:
    """not exact.
    see https://github.com/python/cpython/blob/master/Python/compile.c#L860"""
    if ins.opname in ('SETUP_EXCEPT', 'SETUP_FINALLY', 'POP_EXCEPT', 'END_FINALLY'):
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

#
# def __str__(ins: Instruction):
#     return "BCode(opname='{0.opname}', opcode={0.opcode}, arg={0.arg}, argval={0.argval}, argrepr={1}, " \
#            "offset={0.offset}, is_jump_target={0.is_jump_target})".format(
#         self, repr(self.argrepr))


def calculate_stack_depth(cfg: Cfg) -> dict[int, int]:
    """The stack depth is supposed to be independent of path, so dijkstra on the undirected graph suffices
    (and may be too strong, since we don't need minimality).
    We do it bidirectionally because we want to work with unreachable code too.
    """
    res: dict[int, int] = {}
    backwards_cfg = cfg.reverse(copy=True)
    for label in cfg.nodes:
        if cfg[label] and is_return(cfg[label][-1]):
            # add 1 since return also pops
            dijkstra = gu.single_source_dijkstra_path_length(backwards_cfg, source=label, weight='stack_effect')
            res.update({k: v+1 for k, v in dijkstra.items()})
    res.update(gu.single_source_dijkstra_path_length(cfg, source=0, weight='stack_effect'))
    return res


def make_instruction_block_cfg(instructions: Iterable[Instruction]) -> tuple[dict[int, int], Cfg]:
    dbs = {ins.offset: ins for ins in instructions}
    edges = [(b.offset, dbs[j].offset, {'stack_effect': stack_effect})
             for b in dbs.values()
             for (j, stack_effect) in next_list(b) if dbs.get(j) is not None]
    cfg = gu.Cfg(edges, blocks={k: [v] for k, v in dbs.items()})
    depths = calculate_stack_depth(cfg)
    # each node will hold a block of dictionaries - instruction and stack_depth
    return depths, cfg


def make_instruction_block_cfg_from_function(f) -> tuple[dict[int, int], Cfg]:
    return make_instruction_block_cfg(dis.get_instructions(f))
