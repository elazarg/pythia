from collections import namedtuple
import itertools as it
# TAC is an instruction that knows where it stands :)
# i.e, it knows the current stack depth
# I call the stack depth 'tos', although it usually means "the value at the top of the stack"
def test():
    import code_examples
    import cfg
    cfg = cfg.make_graph(code_examples.CreateScene)
    print_3addr(cfg)
    # draw(cfg)


def var(x):
    return chr(ord('α') + x)


def print_3addr(cfg):
    for n in sorted(cfg.node):
        block = cfg[n]['block']
        for ins in block:
            cmd = ins.fmt.format(**ins._asdict())
            print(n, ':\t', cmd , '\t\t', '' and ins)

Tac = namedtuple('Tac', ('gens', 'uses', 'fmt'))
NOP = Tac(gens=(), uses=(), fmt='NOP')

class Ret(Tac): pass
class Raise(Tac): pass
class Yield(Tac): pass

Del = namedtuple('Del', Tac._fields + ('kills',))

Assign = namedtuple('Assign', Tac._fields + ('is_id',))
Import = namedtuple('Import', Tac._fields)

Binary = namedtuple('Binary', Tac._fields + ('op',))
Inplace = namedtuple('Inplace', Tac._fields + ('op',))
Call = namedtuple('Call', Tac._fields + ('func', 'args', 'kwargs',))

Jump = namedtuple('Jump', Tac._fields + ('target',))
For = namedtuple('For', Tac._fields + ('target',))


def delete(*vs):
    return Del(kills=set(vs), gens=(), uses=(), fmt='DEL {kills}')

def unary(lhs, op):
    return call(lhs, '{}.{}'.format(lhs, op))

def assign(lhs, rhs, is_id=True):
    return Assign(gens=[lhs], uses=[rhs], is_id=is_id,
                  fmt='{gens[0]} = {uses[0]}')

def assign_attr(lhs, rhs, attr, is_id=True):
    return Assign(gens=[lhs], uses=[rhs], is_id=is_id,
                  fmt='{gens[0]} = {uses[0]}.'+attr)

def mulassign(*lhs, rhs, is_id=True):
    return Assign(gens=lhs, uses=[rhs], is_id=is_id,
                  fmt=', '.join('gens[' + i + ']' for i in range(lhs)) + ' = {uses[0]}')

def binary(lhs, left, op, right):
    # note that operators are not *exactly* like attribute access, since it is never an attribute
    return Binary(gens=[lhs], uses=(left, right), op=op,
               fmt='{gens[0]} = {uses[0]} {op} {uses[1]}')

def inplace(lhs, op, rhs):
    return Inplace(gens=(), uses=(lhs, rhs), op=op,
               fmt='{uses[0]} {op}= {uses[1]}')

def call(lhs, f, args=(), kwargs=()):
    fmt_args = ', '.join('{uses[' + str(x) + ']}' for x in range(1, len(args) + 1))
    fmt_kwargs = ', '.join('{uses[' + str(x) + ']:uses[' + str(x + 1) + ']:}' for x in range(len(args) + 1, len(args) + 1 + (len(kwargs) // 2), 2))
    return Call(gens=[lhs], uses=[f, *args, *kwargs], func=f, args=args, kwargs=kwargs,
                fmt='{gens[0]} = {uses[0]}(' + fmt_args + ')' + \
                     (('(kw=' + fmt_kwargs + ')') if kwargs else ''))

def foreach(lhs, iterator, val):
    return For(gens=[lhs], uses=[iterator], target=val,
               fmt='{gens[0]} = next({uses[0]}) HANDLE: GOTO {target}')

def include(lhs, modname):
    return Import(gens=[lhs], uses=(),
                  fmt='{gens[0]} = IMPORT ' + modname)

def include1(lhs, feature, modname):
    return Import(gens=[lhs], uses=(),
                fmt=('{gens[0]} = FROM ' + modname + ' IMPORT ' + feature))


def get_gens(block):
    return set(it.chain.from_iterable(ins.gens for ins in block))

def get_uses(block):
    return set(it.chain.from_iterable(ins.uses for ins in block))

def make_TAC(opname, val, stack_effect, tos):
    tac = make_TAC_no_dels(opname, val, stack_effect, tos)
    #this is simplistic analysis, that is not correct in general:
    tac = list(map(delete, get_gens(tac)-get_uses(tac))) + tac
    if stack_effect < 0:
        tac += [delete(var(tos + i)) for i in range(stack_effect + 1, 1)]
    return tac

def make_TAC_no_dels(opname, val, stack_effect, tos):
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
        return [] # will be completed by make_TAC
    elif name == 'DELETE_FAST':
        return [] # will be completed by make_TAC
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
        return [Ret(gens=(), uses=[var(tos)], fmt='RETURN {uses[0]}')]
    elif name == 'YIELD_VALUE':
        return [Yield(gens=[var(out)], uses=[var(tos)],
                fmt='YIELD {uses[0]}')]
    elif name == 'FOR_ITER':
        return [foreach(var(out), var(tos), val)]
    elif name == 'LOAD':
        if op == 'ATTR':
            return [call(var(out), 'BUILTINS.getattr', [var(tos), val])]
        if op == 'FAST':     rhs = val
        elif op == 'CONST':  rhs = repr(val)
        elif op == 'DEREF':  rhs = 'NONLOCAL.{}'.format(val)
        elif op == 'GLOBAL': rhs = 'GLOBALS.{}'.format(val)
        return [assign(var(out), rhs, is_id=(op != 'CONST'))]
    elif name.startswith('STORE_FAST'):
        return [assign(val, var(tos))]
    elif name == 'STORE_GLOBAL':
        return [assign('GLOBALS.{}'.format(val), var(tos))]
    elif name.startswith('STORE_ATTR'):
        return [assign('{}.{}'.format(var(tos), val), var(tos - 1))]
    elif name.startswith('STORE_SUBSCR'):
        return [call(var(tos), 'BUILTINS.getattr', [var(tos-1), "'__setitem__'"]),
                call(var(tos), var(tos), [var(tos - 2)])]
    elif name == 'BINARY_SUBSCR':
        return [call(var(out), 'BUILTINS.getattr', [var(tos-1), "'__getitem__'"]),
                call(var(out), var(out), [var(tos)])]
    elif name == 'POP_BLOCK':
        return [NOP]
    elif name == 'SETUP_LOOP':
        return [NOP]
    elif name == 'RAISE_VARARGS':
        return [Raise(gens=(), uses=[var(tos)], fmt='RAISE {uses[0]}')]
    elif name == 'UNPACK_SEQUENCE':
        seq = [var(tos + i) for i in reversed(range(val))]
        return [mulassign(*seq, rhs=var(tos))]
    elif name == 'IMPORT_NAME':
        return [include(var(out), val)]
    elif name == 'IMPORT_FROM':
        return [include1(var(out), val, var(tos))]
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
        mid = [var(i + 1) for i in range(tos - nkwargs - nargs, tos - nkwargs)]
        mid_kw = [(var(i) + ': ' + var(i + 1))
                            for i in range(tos - nkwargs + 1, tos, 2)]
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
