from collections import namedtuple
import itertools as it
from enum import Enum

# TAC: Three Address Code. An instruction that knows where it stands.
# i.e, it knows the current stack depth
# I call the stack depth 'tos', although in Python docs it means "the value at the top of the stack"

BLOCKNAME = 'tac_block'

def test():
    import code_examples
    cfg = make_tacblock_cfg(code_examples.CreateScene)
    print_3addr(cfg)
    # draw(cfg)


def print_3addr(cfg):
    for n in sorted(cfg.nodes()):
        # The call for sorted() gives us for free the ordering of blocks by "fallthrough"
        # It is not guaranteed anywhere, but it makes sense - the order of blocks is the 
        # order of the lines in the code
        block = cfg.node[n][BLOCKNAME]
        for ins in block:
            cmd = ins.fmt.format(**ins._asdict())
            print(n, ':\t', cmd)


def make_tacblock_cfg(f):
    def bcode_block_to_tac_block(n, block_data):
        return list(it.chain.from_iterable(
                    make_TAC(bc.opname, bc.argval, bc.stack_effect(), tos, bc.starts_line)
                    for bc, tos in bcode_cfg.get_code_depth_pairs(block_data)))
    
    import bcode_cfg
    bcode_blocks = bcode_cfg.make_bcode_block_cfg_from_function(f)
    import graph_utils as gu
    tac_blocks = gu.node_data_map(bcode_blocks, 
                                  bcode_block_to_tac_block,
                                  BLOCKNAME)
    return tac_blocks


def var(x):
    return '@v{}'.format(x - 1)

def is_stackvar(v):
    return v[0] == '@'

class OP(Enum):
    NOP = 0
    ASSIGN = 1
    IMPORT = 2
    BINARY = 3
    INPLACE = 4
    CALL = 5
    JUMP = 6
    FOR = 7
    RET = 8
    RAISE = 9
    YIELD = 10
    DEL = 11
    
# Tac is most of the interface of this module:
    
fields = ('opcode', 'fmt', 'gens', 'uses', 'kills',
          'is_id', 'op', 'target',
          'func', 'args', 'kwargs',
          'starts_line')

class Tac(namedtuple('Tac', fields)):
    __slots__ = ()
    @property
    def is_assign(self):
        return self.opcode == OP.ASSIGN 

    @property
    def is_del(self):
        return self.opcode == OP.DEL 

    @property
    def is_inplace(self):
        return self.opcode == OP.INPLACE
    
    def format(self):
        return self.fmt.format(**self._asdict()) \
            # +  ' \t\t -- kills: {}'.format(self.kills) 

def tac(opcode, *, fmt, gens=(), uses=(), kills=(), is_id=False, op=None,
        target=None, func=None, args=(), kwargs=(), starts_line=None):
    assert isinstance(opcode, OP)
    assert isinstance(fmt, str)
    for args in [uses, kills, args]:
        assert isinstance(args, (tuple, frozenset))
    tac = Tac(opcode, fmt=fmt, gens=gens, uses=uses, kills=kills or gens, is_id=is_id,
               op=op, target=target, func=func, args=args, kwargs=kwargs,
               starts_line=starts_line)
    return tac

NOP = tac(OP.NOP, fmt='NOP')

def delete(*vs):
    return tac(OP.DEL, kills=vs,
               fmt='DEL {kills}')

def unary(lhs, op):
    return call(lhs, '{}.{}'.format(lhs, op))

def assign(lhs, rhs, is_id=True):
    return tac(OP.ASSIGN, gens=(lhs,), uses=(rhs,), is_id=is_id,
                  fmt='{gens[0]} = {uses[0]}')

def assign_attr(lhs, rhs, attr, is_id=True):
    return tac(OP.ASSIGN, gens=(lhs,), uses=(rhs,), is_id=is_id,
                  fmt='{gens[0]} = {uses[0]}.' + attr)

def mulassign(*lhs, rhs, is_id=True):
    return tac(OP.ASSIGN, gens=lhs, uses=(rhs,), is_id=is_id,
                  fmt=', '.join('gens[{}]'.format(i) for i in range(lhs)) + ' = {uses[0]}')

def binary(lhs, left, op, right):
    # note that operators are not *exactly* like attribute access, since it is never an attribute
    return tac(OP.BINARY, gens=(lhs,), uses=(left, right), op=op,
               fmt='{gens[0]} = {uses[0]} {op} {uses[1]}')

def inplace(lhs, rhs, op):
    return tac(OP.INPLACE, uses=(lhs, rhs), gens=(lhs,), op=op,
               fmt='{uses[0]} {op}= {uses[1]}')

def call(lhs, f, args=(), kwargs=()):
    # this way of formatting wont let use change the number of arguments easily.
    # but it is unlikely to be needed  
    fmt_args = ', '.join('{{uses[{}]}}'.format(x) for x in range(1, len(args) + 1))
    fmt_kwargs = ', '.join('{{uses[{}]:uses[{}]:}}'.format(x, x + 1)
                           for x in range(len(args) + 1, len(args) + 1 + (len(kwargs) // 2), 2))
    return tac(OP.CALL, gens=(lhs,), uses=((f,) + args + kwargs), func=f, args=args, kwargs=kwargs,
                fmt='{gens[0]} = {uses[0]}(' + fmt_args + ')' + \
                     (('(kw=' + fmt_kwargs + ')') if kwargs else ''))

def foreach(lhs, iterator, target):
    return tac(OP.FOR, gens=(lhs,), uses=(iterator,), target=target,
               fmt='{gens[0]} = next({uses[0]}) HANDLE: GOTO {target}')

def jump(target, cond='True'):
    return tac(OP.JUMP, uses=(cond,), target=target,
               fmt='IF {uses[0]} GOTO {target}')

def include(lhs, modname, feature=None):
    fmt = '{gens[0]} = IMPORT(' + modname + ')'
    if feature is not None:
        fmt += '.' + feature 
    return tac(OP.IMPORT, gens=(lhs,), fm=fmt)

def get_gens(block):
    return set(it.chain.from_iterable(ins.gens for ins in block))

def get_uses(block):
    return set(it.chain.from_iterable(ins.uses for ins in block))

def make_TAC(opname, val, stack_effect, tos, starts_line=None):
    tac = make_TAC_no_dels(opname, val, stack_effect, tos)
    # this is simplistic klls analysis, that is not correct in general:
    tac = list(map(delete, get_gens(tac) - get_uses(tac))) + tac
    if stack_effect < 0:
        tac += [delete(var(tos + i)) for i in range(stack_effect + 1, 1)]
    return [t._replace(starts_line=starts_line) for t in tac]


def make_TAC_no_dels(opname, val, stack_effect, tos):
    '''Yes, this is a long function, since it partially describes the semantics of the bytecode,
    So it is a long switch. Similar to the one seen in interpreters.
    It is likely that a table-driven approach will be cleaner and more portable.
    '''
    out = tos + stack_effect if tos is not None else None
    name, op = choose_category(opname, val)
    if name == 'UNARY_ATTR':
        return [unary(var(tos), op)]
    if name == 'UNARY_FUNC':
        return [call(var(out), op, (var(tos),))]
    elif name == 'BINARY':
        return [binary(var(out), var(tos - 1), op, var(tos))]
    elif name == 'INPLACE':
        return [inplace(var(out), var(tos), op)]
    elif name.startswith('POP_JUMP_IF_'):
        res = [call(var(tos), 'not', (var(tos),))] if name.endswith('FALSE') else [] 
        return res + [jump(val, var(tos))]
    elif name == 'JUMP':
        return [jump(val)]
    elif name == 'POP_TOP':
        return []  # will be completed by the dels in make_TAC
    elif name == 'DELETE_FAST':
        return []  # will be completed by the dels in make_TAC
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
        return [tac(OP.RET, uses=(var(tos),), fmt='RETURN {uses[0]}')]
    elif name == 'YIELD_VALUE':
        return [tac(OP.YIELD, gens=[var(out)], uses=[var(tos)],
                fmt='YIELD {uses[0]}')]
    elif name == 'FOR_ITER':
        return [foreach(var(out), var(tos), val)]
    elif name == 'LOAD':
        if op == 'ATTR':
            return [call(var(out), 'BUILTINS.getattr', (var(tos), val))]
        if op in ['FAST', 'NAME']:     rhs = val
        elif op == 'CONST':  rhs = repr(val)
        elif op == 'DEREF':  rhs = 'NONLOCAL.{}'.format(val)
        elif op == 'GLOBAL': rhs = 'GLOBALS.{}'.format(val)
        return [assign(var(out), rhs, is_id=(op != 'CONST'))]
    elif name in ['STORE_FAST', 'STORE_NAME']:
        return [assign(val, var(tos))]
    elif name == 'STORE_GLOBAL':
        return [assign('GLOBALS.{}'.format(val), var(tos))]
    elif name == 'STORE_ATTR':
        return [call(var(tos), 'setattr', (var(tos), repr(val), var(tos-1)))]
    elif name.startswith('STORE_SUBSCR'):
        return [call(var(tos), 'BUILTINS.getattr', (var(tos - 1), "'__setitem__'")),
                call(var(tos), var(tos), (var(tos - 2),))]
    elif name == 'BINARY_SUBSCR':
        return [call(var(out), 'BUILTINS.getattr', (var(tos - 1), "'__getitem__'")),
                call(var(out), var(out), (var(tos),))]
    elif name == 'POP_BLOCK':
        return [NOP]
    elif name == 'SETUP_LOOP':
        return [NOP]
    elif name == 'RAISE_VARARGS':
        return [tac(OP.RAISE, gens=(), uses=[var(tos)], fmt='RAISE {uses[0]}')]
    elif name == 'UNPACK_SEQUENCE':
        seq = [var(tos + i) for i in reversed(range(val))]
        return [mulassign(*seq, rhs=var(tos))]
    elif name == 'IMPORT_NAME':
        return [include(var(out), val)]
    elif name == 'IMPORT_FROM':
        return [include(var(out), var(tos), val)]
    elif name == 'BUILD':
        if op == 'SLICE':
            if val == 2:
                args = (var(tos - 1), var(tos))
            else:
                args = (var(tos), var(tos - 1), var(tos - 2))
            return [call(var(out), 'SLICE', args)]
        return [call(var(out), op, tuple(var(i + 1) for i in range(tos - val, tos)))]
    elif name == 'CALL_FUNCTION':
        nargs = val & 0xFF
        nkwargs = 2 * ((val >> 8) & 0xFF)
        total = nargs + nkwargs
        mid = [var(i + 1) for i in range(tos - nkwargs - nargs, tos - nkwargs)]
        mid_kw = [(var(i) + ': ' + var(i + 1))
                            for i in range(tos - nkwargs + 1, tos, 2)]
        return [call(var(out), var(tos - total), tuple(mid), tuple(mid_kw))]
    assert False, 'Falling through. {}: {}'.format(opname, val)



def choose_category(opname, argval):
    # I'm not sure that this function is actually helpful. (ELAZAR)
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
'OR':    '|'
}

# TODO. == to __eq__, etc.
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
