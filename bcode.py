import dis, inspect
import code_examples

class BCode(dis.Instruction):    
    @property
    def is_sequencer(self):
        return 'RETURN' in self.opname \
            or 'CONTINUE' in self.opname \
            or 'BREAK' in self.opname \
            or self.opname in ['JUMP_FORWARD', 'JUMP_ABSOLUTE', 'END_FINALLY']
            
    @property
    def is_jump_source(self):
        return self.is_jump or self.is_sequencer or self is OP_START
        # yield does not interrupt the flow
        # exceptions do not count, since they can occur practically anywhere
            
    @property
    def is_jump(self):
        return self.opcode in dis.hasjrel + dis.hasjabs
               
    @property
    def size(self):
        return 1 if self.opcode < dis.HAVE_ARGUMENT else 3
        
    def next_list(self):
        # semantics of BREAK is too complicated, in the presence of `else`
        #assert self.opname != 'BREAK_LOOP'
         
        next_offset = self.offset + self.size
        if self.opname == 'FOR_ITER':
            return [(next_offset, self.stack_effect()),
                    (self.argval, -1)]
        res = []
        if not self.is_sequencer:
            res.append( (next_offset, self.stack_effect() ) )
        if self.is_jump_source and self is not OP_START:
            res.append( (self.argval, self.stack_effect()) )
        return res
    
    @property 
    def is_block_boundary(self):
        return self.is_jump_source or self.is_jump_target
             
    def stack_effect(self):
        '''not exact.
        see https://github.com/python/cpython/blob/master/Python/compile.c#L860'''
        if self is OP_START: return 0
        if self.opname in ('SETUP_EXCEPT', 'SETUP_FINALLY', 'POP_EXCEPT', 'RAISE_VARARGS', 'END_FINALLY'):
            assert False, 'for all we know. we assume no exceptions'
        if self.opname == 'BREAK_LOOP' and self.argrepr.startswith('FOR'):
            return -1
        return dis.stack_effect(self.opcode, self.arg) 
     
    @property
    def is_for_iter(self):
        return self.opname == 'FOR_ITER'

def source(f):
    return inspect.getsource(f)

def update_break_instruction(bcodes):
    '''BREAK_LOOP is problematic, since it does not contain the target
    and since the stack effect depends on whether it is inside FOR or not
    the right way to fix it is from inside the graph, but for most purposes
    running over the code will suffice.'''
    s = []
    for i, r in enumerate(bcodes):
        if r.opname == 'SETUP_EXCEPT':
            s.append( None )
        if r.opname =='SETUP_LOOP':
            s.append( [r, 'WHILE '] )
        if r.opname == 'POP_BLOCK':
            s.pop()
        if r.opname == 'FOR_ITER':
            s[-1][-1] = 'FOR '
        if r.opname == 'BREAK_LOOP':
            to =  s[-1][0].argval
            last = next(x for x in reversed(s) if x is not None)
            bcodes[i] = BCode(*(*r[:3], to, '{}to {}'.format(last[-1], to), *r[5:])) 
    assert not s

def get_instructions(f):
    res =[OP_START] + [BCode(*i) for i in dis.get_instructions(f)]
    update_break_instruction(res)
    # res.append(make_end(next(res[-1].next_list)))
    return res

OP_START = BCode(opname='START', opcode=-1, arg=None, argval=0, argrepr=None, offset=-1, starts_line=-1, is_jump_target=False)

def make_end(offset):
    return BCode(opname='END', opcode=-2, arg=None, argval=None, argrepr=None, offset=offset, starts_line=None, is_jump_target=False)



def example():
    x = 1
    for x in range(7):
        for y in [1,2,3]:
            print(y)
    else:
        len([])

if __name__ == '__main__':   
    for b in get_instructions(code_examples.mb_to_tkinter):
        print(b)

BIN_TO_OP = {
'POWER':    '**', 
'MULTIPLY':    '*', 
'MATRIX_MULTIPLY':    '@', 
'FLOOR_DIVIDE':    '//', 
'TRUE_DIVIDE':    '/', 
'MODULO':    '%', 
'ADD':    '+', 
'SUBTRACT': '-', 
'SUBSCR':   '[]',
'LSHIFT':    '<<' ,
'RSHIFT':    '>>', 
'AND':    '&', 
'XOR':    '^', 
'OR':    '|', 
}

UN_TO_OP = {
'UNARY_POSITIVE' : '+',
'UNARY_NEGATIVE' : '-',
'UNARY_NOT': 'not ',
'UNARY_INVERT': '~',
}