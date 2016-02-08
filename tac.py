from collections import namedtuple
# TAC is and instruction that knows where it stands :)
# i.e, it knows the current stack depth
# I call the stack depth 'tos', although it usually means "the value at the top of the stack"


class TAC(namedtuple('TAC', ('category', 'op', 'argval', 'ins', 'tos', 'out'))):
    def get_defs(self):
        if self.category in ['UNARY', 'BINARY', 'DUP_TOP', 'BUILD',
                             'LOAD', 'STORE_FAST', 'CALL_FUNCTION']:
            return (var(self.out),)
        if 'IMPORT' in self.category:
            return (var(self.out),)
        if self.category == 'UNPACK_SEQUENCE':
            return tuple(var(self.tos + i) for i in reversed(range(self.val)))

def var(x):
    return chr(ord('α') + x)


def test():
    import code_examples
    import cfg
    cfg = cfg.make_graph(code_examples.CreatePlasmaCube)
    print_3addr(cfg)
    # draw(cfg)


def print_3addr(cfg):
    for n in sorted(cfg.node):
        block = cfg[n]['block']
        for ins in block:
            cmd = ins.fmt.format(**ins._asdict())
            print(n, ':\t', cmd , '\t\t', '' and ins)


Binary = namedtuple('Binary', ('defs', 'uses', 'kills', 'op', 'fmt'))
Inplace = namedtuple('Inplace', ('defs', 'uses', 'kills', 'op', 'fmt'))
Assign = namedtuple('Assign', ('defs', 'uses', 'kills', 'fmt'))
Call = namedtuple('Call', ('defs', 'func', 'uses', 'args', 'kwargs', 'fmt'))
Jump = namedtuple('Jump', ('uses', 'target', 'fmt'))

For = namedtuple('For', ('defs', 'uses', 'target', 'fmt'))
Ret = namedtuple('Ret', ('uses', 'fmt'))
Raise = namedtuple('Raise', ('uses', 'fmt'))
Yield = namedtuple('Yield', ('defs', 'uses', 'fmt'))
Del = namedtuple('Del', ('kills', 'fmt'))
NOP = namedtuple('Nop', ('fmt'))(fmt='NOP')

def delete(v):
    return Del(kills=[v], fmt='DEL {kills[0]}')

def unary(lhs, op):
    return call(lhs, '{}.{}'.format(lhs, op))

def assign(lhs, rhs):
    return Assign(defs=[lhs], uses=[rhs], kills=[],
                  fmt='{defs[0]} = {uses[0]}')

def mulassign(*lhs, rhs):
    return Assign(defs=lhs, uses=[rhs], kills=[],
                  fmt=', '.join('defs[' + i + ']' for i in range(lhs)) + ' = {uses[0]}')

def binary(lhs, left, op, right):
    return Binary(defs=[lhs], uses=[left, right], kills=[], op=op,
               fmt='{defs[0]} = {uses[0]} {op} {uses[1]}')

def inplace(lhs, op, rhs):
    return Inplace(defs=[], uses=[lhs, rhs], kills=[], op=op,
               fmt='{uses[0]} {op}= {uses[1]}')
    
def call(lhs, f, args=(), kwargs=()):
    return Call(defs=[lhs], uses=[f, *args, *kwargs], func=f, args=args, kwargs=kwargs,
                fmt='{defs[0]} = {func}(args={args}, kw={kwargs})')

def foreach(lhs, iterator, val):
    return For(defs=[lhs], uses=[iterator], target=val,
               fmt='{defs[0]} = next({uses[0]}) HANDLE: GOTO {target}')

def make_TAC(opname, val, stack_effect, tos):
    out = tos + stack_effect if tos is not None else None
    name, op = choose_category(opname, val)
    if name == 'UNARY_ATTR':
        return [unary(var(tos), op)]
    if name == 'UNARY_FUNC':
        return [call(var(out), op, var(tos))]
    elif name == 'BINARY':
        return [binary(var(out), var(tos - 1), op, var(tos))]
    elif name == 'INPLACE':
        return [inplace(var(out), var(tos), op)]
    elif name == 'JUMP':
        return [Jump(uses=[True], target=val,
                   fmt='IF {uses[0]} GOTO {target}')]
    elif name.startswith('POP_JUMP_IF_'):
        res = [call(var(tos), 'not', var(tos))] if name.endswith('FALSE') else [] 
        return res + [Jump(uses=[var(tos)], target=val,
                    fmt='IF {uses[0]} GOTO {target}')]
    elif name == 'POP_TOP':
        return [delete(var(tos))]
    elif name == 'DELETE_FAST':
        return [delete(val)]
    elif name == 'ROT_TWO':
        fresh = var(tos + 1)
        return [assign(fresh, var(tos)),
                assign(var(tos), var(tos - 1)),
                assign(var(tos - 1), fresh),
                delete(fresh)]
    elif name == 'ROT_THREE':
        fresh = var(tos + 1)
        return [assign(fresh, var(tos - 2)),
                assign(var(tos - 2), var(tos - 1)),
                assign(var(tos - 1), var(tos)),
                assign(var(tos), fresh),
                delete(fresh)]
    elif name == 'DUP_TOP':
        return [assign(var(out), var(tos))]
    elif name == 'DUP_TOP_TWO':
        return [assign(var(out), var(tos - 2)),
                assign(var(out + 1), var(tos - 1))]
    elif name == 'RETURN_VALUE':
        return [Ret(uses=[var(tos)], fmt='RETURN {uses[0]}')]
    elif name == 'YIELD_VALUE':
        return [Yield(defs=[var(out)], uses=[var(tos)],
                fmt='YIELD {uses[0]}')]
    elif name == 'FOR_ITER':
        return [foreach(var(out), var(tos), val)]
    elif name == 'LOAD':
        if op == 'FAST':     rhs = val
        elif op == 'CONST':  rhs = repr(val)
        elif op == 'ATTR':   rhs = '{}.{}'.format(var(tos), val)
        elif op == 'DEREF':  rhs = 'NONLOCAL.{}'.format(val)
        elif op == 'GLOBAL': rhs = 'GLOBALS.{}'.format(val)
        return [assign(var(out), rhs)]
    elif name.startswith('STORE_FAST'):
        return [assign(val, var(tos))]
    elif name == 'STORE_GLOBAL':
        return [assign('GLOBALS.{}'.format(val), var(tos))]
    elif name.startswith('STORE_ATTR'):
        return [assign('{}.{}'.format(var(tos), val), var(tos - 1))]
    elif name.startswith('STORE_SUBSCR'):
        return [call(var(tos), '{}.__setitem__'.format(tos - 1), var(tos - 2))]
    elif name == 'BINARY_SUBSCR':
        return [call(var(out), var(tos - 1) + '.__getitem__', [var(tos)])]
    elif name == 'POP_BLOCK':
        return [NOP]
    elif name == 'SETUP_LOOP':
        return [NOP]
    elif name == 'RAISE_VARARGS':
        return [Raise(uses=[var(tos)], fmt='RAISE {uses[0]}')]
    elif name == 'UNPACK_SEQUENCE':
        seq = [var(tos + i) for i in reversed(range(val))]
        return [mulassign(*seq, rhs=var(tos))]
    elif name == 'IMPORT_NAME':
        return ['{} = IMPORT {}'.format(var(out), val)]
    elif name == 'IMPORT_FROM':
        return ['{} = FROM {} IMPORT {}'.format(var(out), var(tos), val)]
    elif name == 'BUILD':
        if op == 'SLICE':
            if val == 2:
                args = [var(tos - 1), var(tos)]
            else:
                args = [var(tos), var(tos - 1), var(tos - 2)]
            return [call(var(out), 'SLICE', args)]
        return [call(var(out), op, [var(i + 1) for i in range(tos - val, tos)])]
    elif name == 'CALL_FUNCTION':
        nargs = val & 0xFF
        nkwargs = 2 * ((val >> 8) & 0xFF)
        total = nargs + nkwargs
        mid = ', '.join(var(i + 1)
                    for i in range(tos - nkwargs - nargs, tos - nkwargs))
        mid_kw = ', '.join((var(i) + ': ' + var(i + 1))
                            for i in range(tos - nkwargs + 1, tos, 2))
        return [call(var(out), var(tos - total), mid, mid_kw)]
    assert False, '{}: {}'.format(opname, val)



def choose_category(opname, argval):
    if opname in ('UNARY_POSITIVE', 'UNARY_NEGATIVE', 'UNARY_INVERT'):
        return 'UNARY_ATTR', UN_TO_OP[opname]
    if opname in ('GET_ITER', 'UNARY_NOT', 'GET_YIELD_FROM_ITER'):
        return 'UNARY_FUNC', UN_TO_OP[opname]
               
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

CMPOP_TO_OP = {
               
}

UN_TO_OP = {
            # __abs__ ?
'UNARY_POSITIVE' : '__pos__',
'UNARY_NEGATIVE' : '__neg__',
'UNARY_NOT': 'not ',
'UNARY_INVERT': '__invert__',
'GET_ITER':'ITER ',
'GET_YIELD_FROM_ITER': 'YIELD_FROM_ITER '
}

if __name__ == '__main__':   
    test()
