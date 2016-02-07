from collections import namedtuple
# TAC is and instruction that knows where it stands :)
# i.e, it knows the current stack depth
# I call the stack depth 'tos', although it usually means "the value at the top of the stack"


class TAC(namedtuple('TAC', ('category', 'op', 'argval', 'ins', 'tos', 'out'))):
    def get_defs(self):
        if self.category in ['UNARY', 'BINARY', 'DUP_TOP', 'BUILD',
                             'LOAD', 'STORE_FAST', 'CALL_FUNCTION']:
            return (var(self.out), )
        if 'IMPORT' in self.category:
            return (var(self.out), )
        if self.category == 'UNPACK_SEQUENCE':
            return tuple(var(self.tos+i) for i in reversed(range(self.val)))

def var(x):
    return chr(ord('α') + x)

def make_TAC(bcode, tos):
    out = tos + bcode.stack_effect() if tos is not None else None
    category, op = choose_category(bcode.opname, bcode.argval)
    return TAC(category=category, op=op, argval=bcode.argval,
               tos=tos, out=out, ins=bcode)


def test():
    import code_examples
    import cfg
    cfg = cfg.make_graph(code_examples.example)
    print_3addr(cfg)
    #draw(cfg)


def print_3addr(cfg):
    for n in sorted(cfg.node):
        block = cfg[n]['block']
        if block[0].tos is None:
            # unreachable code
            print('UNREACHABLE:', block)
            continue
        for ins in block:
            cmd = to3addr(ins)
            print(n,':\t', cmd , '\t\t', '' and ins)


def choose_category(opname, argval):
    if opname in ('UNARY_POSITIVE', 'UNARY_NEGATIVE', 'UNARY_NOT', 'UNARY_INVERT', 'GET_ITER', 'GET_YIELD_FROM_ITER'):
        return 'UNARY', UN_TO_OP[opname]
       
    if opname in ('JUMP_ABSOLUTE', 'JUMP_FORWARD', 'BREAK_LOOP', 'CONTINUE_LOOP'):
        return 'JUMP', None
    
    desc = opname.split('_', 1)[1]
    if opname == 'COMPARE_OP':
        return 'BINARY', argval
    if opname.startswith('BINARY_') and opname != 'BINARY_SUBSCR':
        return 'BINARY', BIN_TO_OP[desc]

    if opname.startswith('INPLACE_'):
        return 'INPLACE', BIN_TO_OP[desc]
    
    if opname.startswith('BUILD_'):
        return 'BUILD', desc
 
    if opname.startswith('LOAD_'):
        return 'LOAD', desc

    return opname, None


def to3addr(ins:'TAC'):
    tos = ins.tos
    out = ins.out
    name = ins.category
    op = ins.op
    val = ins.argval
    if name == 'UNARY':
        return '{0} = {1}{0}'.format(var(tos), op)
    elif name == 'BINARY':
        return '{} = {} {} {}'.format(var(out), var(tos - 1), op, var(tos))
    elif name == 'INPLACE':
        return '{} {}= {}'.format(var(tos - 1), op, var(tos))
    elif name == 'JUMP':
        return 'GOTO {}'.format(val)
    elif name == 'POP_TOP':
        return 'DEL {}'.format(var(tos))
    elif name == 'ROT_TWO':
        return '{0}, {1} = {1}, {0}'.format(var(tos), var(tos-1))
    elif name == 'ROT_THREE':
        return '{0}, {1}, {2} = {2}, {0}, {1}'.format(var(tos), var(tos-1), var(tos-1))
    elif name == 'DUP_TOP':
        return '{0} = {1}'.format(var(out), var(tos))
    elif name == 'DUP_TOP_TWO':
        return '{0}, {1} = {2}, {3}'.format(var(out), var(tos-2),var(out+1), var(tos-1))
    elif name == 'RETURN_VALUE':
        return 'RETURN {}'.format(var(tos))
    elif name == 'YIELD_VALUE':
        return 'YIELD {}'.format(var(tos))
    elif name.startswith('POP_JUMP_IF_'):
        return 'IF {}{} GOTO {}'.format('' if name.endswith('TRUE') else 'NOT ', 
            var(tos), val)
    elif name == 'BUILD':
        if op == 'SLICE':
            if val == 2:
                fmt = '{0} = SLICE({2}:{1})'
            else:
                fmt = '{0} = SLICE({1}:{2}:{3})'
            return fmt.format(var(out), var(tos), var(tos-1), var(tos-2))
        mid = ', '.join(var(i + 1) for i in range(tos - val, tos))
        return '{} = {}({})'.format(var(out), op, mid)
    elif name == 'FOR_ITER':
        return '{} = next({}) HANDLE: DEL {} and GOTO {}'.format(var(out), var(tos), var(tos), val)
    elif name == 'LOAD':
        if op == 'FAST':
            return '{} = {}'.format(var(out), val)
        elif op == 'DEREF':
            return '{} = NONLOCAL.{}'.format(var(out), val)
        elif op == 'CONST':
            return '{} = {}'.format(var(out), repr(val))
        elif op == 'ATTR':
            return '{} = {}.{}'.format(var(out), var(tos), val)
        elif op == 'GLOBAL':
            return '{} = GLOBALS.{}'.format(var(out), val)
    elif name == 'STORE_GLOBAL':
        return 'GLOBALS.{} = {}'.format(val, var(tos))
    elif name.startswith('STORE_ATTR'):
        return '{}.{} = {}'.format(var(tos), val, var(tos-1))
    elif name.startswith('STORE_SUBSCR'):
        return '{}[{}] = {}'.format(var(tos-1), var(tos), var(tos-2))
    elif name == 'BINARY_SUBSCR':
        return '{0} = {1}[{2}]'.format(var(out), var(tos - 1), var(tos))
    elif name.startswith('STORE_FAST'):
        return '{} = {}'.format(val, var(tos))
    elif name == 'DELETE_FAST':
        return 'DEL {}'.format(val, var(tos))
    elif name == 'POP_BLOCK':
        return 'END LOOP'
    elif name == 'SETUP_LOOP':
        return 'LOOP'
    elif name == 'RAISE_VARARGS':
        return 'raise {}'.format(var(tos))
    elif name == 'UNPACK_SEQUENCE':
        seq = ', '.join(var(tos+i) for i in reversed(range(val)))
        return '{} = {}'.format(seq, var(tos))
    elif name == 'IMPORT_NAME':
        return '{} = IMPORT {}'.format(var(out), val)
    elif name == 'IMPORT_FROM':
        return '{} = FROM {} IMPORT {}'.format(var(out), var(tos), val)
    elif name == 'CALL_FUNCTION':
        nargs = val & 0xFF
        nkwargs = 2*((val >> 8) & 0xFF)
        total = nargs + nkwargs
        mid = ', '.join(var(i + 1)
                    for i in range(tos - nkwargs - nargs, tos - nkwargs))
        mid_kw = ', '.join( (var(i) + ': ' + var(i+1))
                            for i in range(tos - nkwargs + 1, tos, 2))
        return '{} = {}({}, kw=({}))'.format(var(out), var(tos - total), mid, mid_kw)
    else:
        print(ins)
        return '{} = {} {}'.format(var(out), name, var(tos))


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
'SUBSCR':    '[]',  
}


UN_TO_OP = {
'UNARY_POSITIVE' : '+',
'UNARY_NEGATIVE' : '-',
'UNARY_NOT': 'not ',
'UNARY_INVERT': '~',
'GET_ITER':'ITER ',
'GET_YIELD_FROM_ITER': 'YIELD_FROM_ITER '
}

if __name__ == '__main__':   
    test()