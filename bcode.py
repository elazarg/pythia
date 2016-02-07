import dis
import code_examples


class BCode(dis.Instruction):
    __slots__ = ()
    @property    
    def is_sequencer(self):
        return self.opname in sequencers 
            
    @property
    def is_jump_source(self):
        return self.is_jump or self.is_sequencer
        # YIELD_VALUE does not interrupt the flow
        # exceptions do not count, since they can occur practically anywhere
            
    @property
    def is_jump(self):
        return self.opcode in dis.hasjrel \
            or self.opcode in dis.hasjabs
               
    @property
    def size(self):
        return 1 if self.opcode < dis.HAVE_ARGUMENT else 3
    
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
        '''not exact.
        see https://github.com/python/cpython/blob/master/Python/compile.c#L860'''
        if self.opname in ('SETUP_EXCEPT', 'SETUP_FINALLY', 'POP_EXCEPT', 'END_FINALLY'):
            assert False, 'for all we know. we assume no exceptions'
        if self.is_raise:
            # if we wish to analyze exception path, we should break to except: and push 3, or somthing.
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
        return "BCode(opname='{0.opname}', opcode={0.opcode}, arg={0.arg}, argval={0.argval}, argrepr={1}, offset={0.offset}, is_jump_target={0.is_jump_target})".format(
                self, repr(self.argrepr))



def update_break_instruction(ins_iter):
    '''BREAK_LOOP is problematic, since it does not contain the target
    and since the stack effect depends on whether it is inside FOR or not
    the right way to fix it is from inside the graph, but for most purposes
    running over the code will suffice.
    RAISE_VARARGS is obviously problematic too. we want to jump to all `excpet` clauses
    and out of the function.
    '''
    s = []
    for ins in ins_iter:
        if ins.opname == 'SETUP_LOOP':
            s.append([ins, 'WHILE '])
        if ins.opname == 'POP_BLOCK':
            s.pop()
        if ins.opname == 'FOR_ITER':
            s[-1][-1] = 'FOR '
        if ins.opname == 'BREAK_LOOP':
            last = next(x for x in reversed(s))
            to = last[0].argval
            yield BCode(*ins._replace(argrepr='{}to {}'.format(last[-1], to)))
        else:
            yield BCode(*ins)
    assert not s


def get_instructions(f):
    return ((b.offset, b)
            for b in update_break_instruction(dis.get_instructions(f)))

sequencers = frozenset(('RETURN_VALUE', 'CONTINUE_LOOP', 'BREAK_LOOP', 'RAISE_VARARGS', 'JUMP_FORWARD', 'JUMP_ABSOLUTE'))

if __name__ == '__main__':   
    for b in get_instructions(code_examples.getpass):
        print(b)
