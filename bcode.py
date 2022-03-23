import dis
from typing import Iterator


class BCode(dis.Instruction):
    __slots__ = ()

    @property
    def is_sequencer(self):
        return self.opname in (
            'RETURN_VALUE', 'CONTINUE_LOOP', 'BREAK_LOOP', 'RAISE_VARARGS', 'JUMP_FORWARD', 'JUMP_ABSOLUTE')

    @property
    def is_return(self):
        return self.opname == 'RETURN_VALUE'

    @property
    def is_jump_source(self):
        return self.is_jump or self.is_sequencer
        # YIELD_VALUE does not interrupt the flow
        # exceptions do not count, since they can occur practically anywhere

    @property
    def is_jump(self):
        return self.opcode in dis.hasjrel or self.opcode in dis.hasjabs

    # static int
    # instr_size(struct instr *instruction)
    # {
    #     int opcode = instruction->i_opcode;
    #     int oparg = HAS_ARG(opcode) ? instruction->i_oparg : 0;
    #     int extended_args = (0xFFFFFF < oparg) + (0xFFFF < oparg) + (0xFF < oparg);
    #     int caches = _PyOpcode_Caches[opcode];
    #     return extended_args + 1 + caches;
    # }
    @property
    def size(self):
        return 2
        has_arg = self.opcode >= dis.HAVE_ARGUMENT
        oparg = self.arg if has_arg else 0
        extended_args = (0xFFFFFF < oparg) + (0xFFFF < oparg) + (0xFF < oparg)
        return extended_args + 1
        return 1 if self.opcode < dis.HAVE_ARGUMENT else 2

    @property
    def fallthrough_offset(self):
        return self.offset + self.size

    def next_list(self):
        if self.is_raise:
            return []
        if self.is_for_iter:
            return [(self.fallthrough_offset, self.stack_effect()),
                    (self.argval, -1)]
        res = []
        if not self.is_sequencer:
            res.append((self.fallthrough_offset, self.stack_effect()))
        if self.is_jump_source:
            res.append((self.argval, self.stack_effect()))
        return res

    @property
    def is_block_boundary(self):
        return self.is_jump_source or self.is_jump_target

    def stack_effect(self):
        """not exact.
        see https://github.com/python/cpython/blob/master/Python/compile.c#L860"""
        if self.opname in ('SETUP_EXCEPT', 'SETUP_FINALLY', 'POP_EXCEPT', 'END_FINALLY'):
            assert False, 'for all we know. we assume no exceptions'
        if self.is_raise:
            # if we wish to analyze exception path, we should break to except: and push 3, or something.
            return -1
        if self.opname == 'BREAK_LOOP' and self.argrepr.startswith('FOR'):
            return -1
        return dis.stack_effect(self.opcode, self.arg)

    @property
    def is_for_iter(self):
        return self.opname == 'FOR_ITER'

    @property
    def is_raise(self):
        return self.opname == 'RAISE_VARARGS'

    def __str__(self):
        return "BCode(opname='{0.opname}', opcode={0.opcode}, arg={0.arg}, argval={0.argval}, argrepr={1}, " \
               "offset={0.offset}, is_jump_target={0.is_jump_target})".format(
            self, repr(self.argrepr))


def update_break_instruction(ins_iter: Iterator[dis.Instruction]) -> Iterator[BCode]:
    """BREAK_LOOP is problematic:
    (a) It does not contain the target.
    (b) Both stack effect and target depends on whether it is inside FOR or inside WHILE.
    The right way to fix it is from inside the graph, but for most purposes
    running over the code will suffice; It's hard to do so from cfg, since its structure depends on the analysis...
    RAISE_VARARGS is obviously problematic too. We want to jump to all `except` clauses
    and out of the function; but more importantly, its POP_BLOCK should be matched appropriately.
    """
    for ins in ins_iter:
        # if ins.opname == 'SETUP_LOOP':
        #     stack.append([ins, 'WHILE '])
        # if ins.opname == 'POP_BLOCK':
        #     stack.pop()
        # if ins.opname == 'FOR_ITER':
        #     stack[-1][-1] = 'FOR '
        # if ins.opname == 'BREAK_LOOP':
        #     last = stack[-1]
        #     to = last[0].argval
        #     ins = ins._replace(argrepr='{}to {}'.format(last[-1], to))
        yield BCode(*ins)
    # assert len(stack) == 0


def get_instructions(f) -> Iterator[BCode]:
    return update_break_instruction(dis.get_instructions(f))


def test():
    import code_examples
    import pyclbr

    elems = pyclbr.readmodule_ex('code_examples')
    for name, val in elems.items():
        print(val.lineno, ':', name)
    for b in dis.dis(code_examples.simple):
        print(b)


if __name__ == '__main__':
    test()
