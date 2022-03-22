import itertools
import itertools as it
from typing import NamedTuple, Iterable, Optional
from enum import Enum

import bcode_cfg
import bcode
import graph_utils as gu

BLOCKNAME = 'tac_block'
Var = str


def test() -> None:
    import code_examples
    cfg = make_tacblock_cfg(code_examples.simple)
    print_3addr(cfg)
    bcode_cfg.draw(cfg)


def cfg_to_lines(cfg, no_dels=True) -> Iterable[str]:
    for n in sorted(cfg.nodes()):
        # The call for sorted() gives us for free the ordering of blocks by "fallthrough"
        # It is not guaranteed anywhere, but it makes sense - the order of blocks is the 
        # order of the lines in the code
        block = cfg.nodes[n][BLOCKNAME]
        for ins in block:
            if no_dels and ins.opcode == Op.DEL:
                continue
            cmd = ins.fmt.format(**ins._asdict())
            yield '{}:\t{}'.format(n, cmd)


def print_3addr(cfg, no_dels=True) -> None:
    print('\n'.join(cfg_to_lines(cfg, no_dels)))


def make_tacblock_cfg(f):
    depths, bcode_blocks = bcode_cfg.make_bcode_block_cfg_from_function(f)

    def bcode_block_to_tac_block(n, block_data: dict[str, list[dict[str, bcode.BCode]]]) -> list[Tac]:
        block = [bc['BCode'] for bc in block_data['bcode_block']]
        return list(it.chain.from_iterable(
            make_TAC(bc.opname, bc.argval, bc.stack_effect(), depths[bc.offset], bc.starts_line)
            for bc in block))

    tac_blocks = gu.node_data_map(bcode_blocks,
                                  bcode_block_to_tac_block,
                                  BLOCKNAME)
    return tac_blocks


def var(x) -> Var:
    return '$v{}'.format(x - 1)


def is_stackvar(v) -> bool:
    return v[0] == '$'


class Op(Enum):
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
    UNSUPPORTED = 12


class Tac(NamedTuple):
    """Three Address Code. An instruction that knows where it stands.
    i.e, it knows the current stack depth"""
    opcode: Op
    fmt: str
    gens: tuple[Var, ...]
    uses: tuple[Var, ...]
    kills: tuple[Var, ...]
    is_id: bool
    op: Optional[str]
    target: Optional[int]
    func: Optional[int]
    args: tuple[Var]
    kwargs: int
    starts_line: Optional[int]

    @property
    def is_assign(self) -> bool:
        return self.opcode == Op.ASSIGN

    @property
    def is_del(self) -> bool:
        return self.opcode == Op.DEL

    @property
    def is_inplace(self) -> bool:
        return self.opcode == Op.INPLACE

    def format(self) -> str:
        return self.fmt.format(**self._asdict()) \
            # +  ' \t\t -- kills: {}'.format(self.kills) 


def tac(opcode: Op, *, fmt: str, gens: Iterable[Var] = (), uses: Iterable[Var] = (), kills: Iterable[Var] = (),
        is_id=False, op: str = None, target=None, func=None, args: Iterable[Var] = (), kwargs=(),
        starts_line: int = None):
    return Tac(opcode,
               fmt=fmt,
               gens=tuple(gens),
               uses=tuple(uses),
               kills=tuple(kills or gens),
               is_id=is_id,
               op=op,
               target=target,
               func=func,
               args=tuple(args),
               kwargs=kwargs,
               starts_line=starts_line)


NOP = tac(Op.NOP, fmt='NOP')


def delete(*vs) -> Tac:
    return tac(Op.DEL, kills=vs,
               fmt='DEL {kills}')


def unary(lhs: Var, op) -> Tac:
    return call(lhs, '{}.{}'.format(lhs, op))


def assign(lhs: Var, rhs: Var, is_id=True) -> Tac:
    return tac(Op.ASSIGN, gens=(lhs,), uses=(rhs,), is_id=is_id,
               fmt='{gens[0]} = {uses[0]}')


def assign_attr(lhs: Var, rhs: Var, attr, is_id=True) -> Tac:
    return tac(Op.ASSIGN, gens=(lhs,), uses=(rhs,), is_id=is_id,
               fmt='{gens[0]} = {uses[0]}.' + attr)


def mulassign(*lhs: Var, rhs: Var, is_id=True) -> Tac:
    return tac(Op.ASSIGN, gens=lhs, uses=(rhs,), is_id=is_id,
               fmt=', '.join('{{gens[{}]}}'.format(i) for i in range(len(lhs))) + ' = {uses[0]}')


def binary(lhs: Var, left: Var, op, right) -> Tac:
    # note that operators are not *exactly* like attribute access, since it is never an attribute
    return tac(Op.BINARY, gens=(lhs,), uses=(left, right), op=op,
               fmt='{gens[0]} = {uses[0]} {op} {uses[1]}')


def inplace(lhs: Var, rhs: Var, op) -> Tac:
    return tac(Op.INPLACE, uses=(lhs, rhs), gens=(lhs,), op=op,
               fmt='{uses[0]} {op}= {uses[1]}')


def call(lhs: Var, f, args=(), kwargs=()) -> Tac:
    # this way of formatting won't let use change the number of arguments easily.
    # but it is unlikely to be needed  
    fmt_args = ', '.join('{{uses[{}]}}'.format(x) for x in range(1, len(args) + 1))
    fmt_kwargs = ', '.join('{{uses[{}]:uses[{}]:}}'.format(x, x + 1)
                           for x in range(len(args) + 1, len(args) + 1 + (len(kwargs) // 2), 2))
    return tac(Op.CALL, gens=(lhs,), uses=((f,) + args + kwargs), func=f, args=args, kwargs=kwargs,
               fmt='{gens[0]} = {uses[0]}(' + fmt_args + ')' +
                   (('(kw=' + fmt_kwargs + ')') if kwargs else ''))


def foreach(lhs: Var, iterator, target) -> Tac:
    return tac(Op.FOR, gens=(lhs,), uses=(iterator,), target=target,
               fmt='{gens[0]} = next({uses[0]}) HANDLE: GOTO {target}')


def jump(target, cond='True') -> Tac:
    return tac(Op.JUMP, uses=(cond,), target=target,
               fmt='IF {uses[0]} GOTO {target}')


def include(lhs, modname, feature=None) -> Tac:
    fmt = '{gens[0]} = IMPORT(' + modname + ')'
    if feature is not None:
        fmt += '.' + feature
    return tac(Op.IMPORT, gens=(lhs,), fmt=fmt)


def get_gens(block) -> set[Var]:
    return set(it.chain.from_iterable(ins.gens for ins in block))


def get_uses(block) -> set[Var]:
    return set(it.chain.from_iterable(ins.uses for ins in block))


def make_TAC(opname, val, stack_effect, stack_depth, starts_line=None) -> list[Tac]:
    if opname == 'LOAD_CONST' and isinstance(val, tuple):
        lst = []
        for v in val:
            lst += make_TAC('LOAD_CONST', v, 1, stack_depth, starts_line)
            stack_depth += 1
        return lst + make_TAC('BUILD_TUPLE', len(val), -len(val) + 1, stack_depth, starts_line)
    tac = make_TAC_no_dels(opname, val, stack_effect, stack_depth)
    # this is simplistic kills analysis, that is not correct in general:
    tac = list(map(delete, get_gens(tac) - get_uses(tac))) + tac
    if stack_effect < 0:
        tac += [delete(var(stack_depth + i)) for i in range(stack_effect + 1, 1)]
    return [t._replace(starts_line=starts_line) for t in tac]


def make_TAC_no_dels(opname, val, stack_effect, stack_depth) -> list[Tac]:
    """Yes, this is a long function, since it partially describes the semantics of the bytecode,
    So it is a long switch. Similar to the one seen in interpreters.
    It is likely that a table-driven approach will be cleaner and more portable.
    """
    out = stack_depth + stack_effect if stack_depth is not None else None
    name, op = choose_category(opname, val)
    if name == 'UNARY_ATTR':
        return [unary(var(stack_depth), op)]
    elif name == 'UNARY_FUNC':
        return [call(var(out), op, (var(stack_depth),))]
    elif name == 'BINARY':
        return [binary(var(out), var(stack_depth - 1), op, var(stack_depth))]
    elif name == 'INPLACE':
        return [inplace(var(out), var(stack_depth), op)]
    elif name.startswith('POP_JUMP_IF_'):
        res = [call(var(stack_depth), 'not', (var(stack_depth),))] if name.endswith('FALSE') else []
        return res + [jump(val, var(stack_depth))]
    elif name == 'JUMP':
        return [jump(val)]
    elif name == 'POP_TOP':
        return []  # will be completed by the dels in make_TAC
    elif name == 'DELETE_FAST':
        return []  # will be completed by the dels in make_TAC
    elif name == 'ROT_TWO':
        fresh = var(stack_depth + 1)
        return [assign(fresh, var(stack_depth)),
                assign(var(stack_depth), var(stack_depth - 1)),
                assign(var(stack_depth - 1), fresh),
                delete(fresh)]
    elif name == 'ROT_THREE':
        fresh = var(stack_depth + 1)
        return [assign(fresh, var(stack_depth - 2)),
                assign(var(stack_depth - 2), var(stack_depth - 1)),
                assign(var(stack_depth - 1), var(stack_depth)),
                assign(var(stack_depth), fresh),
                delete(fresh)]
    elif name == 'DUP_TOP':
        return [assign(var(out), var(stack_depth))]
    elif name == 'DUP_TOP_TWO':
        return [assign(var(out), var(stack_depth - 2)),
                assign(var(out + 1), var(stack_depth - 1))]
    elif name == 'RETURN_VALUE':
        return [tac(Op.RET, uses=(var(stack_depth),), fmt='RETURN {uses[0]}')]
    elif name == 'YIELD_VALUE':
        return [tac(Op.YIELD, gens=[var(out)], uses=[var(stack_depth)],
                    fmt='YIELD {uses[0]}')]
    elif name == 'FOR_ITER':
        return [foreach(var(out), var(stack_depth), val)]
    elif name == 'LOAD':
        if op == 'ATTR':
            return [call(var(out), 'BUILTINS.getattr', (var(stack_depth), repr(val)))]
        if op in ['FAST', 'NAME']:
            rhs = val
        elif op == 'CONST':
            rhs = repr(val)
        elif op == 'DEREF':
            rhs = 'NONLOCAL.{}'.format(val)
        elif op == 'GLOBAL':
            rhs = 'GLOBALS.{}'.format(val)
        else:
            assert False, op
        return [assign(var(out), rhs, is_id=(op != 'CONST'))]
    elif name in ['STORE_FAST', 'STORE_NAME']:
        return [assign(val, var(stack_depth))]
    elif name == 'STORE_GLOBAL':
        return [assign('GLOBALS.{}'.format(val), var(stack_depth))]
    elif name == 'STORE_ATTR':
        return [call(var(stack_depth), 'setattr', (var(stack_depth), repr(val), var(stack_depth - 1)))]
    elif name.startswith('STORE_SUBSCR'):
        return [call(var(stack_depth), 'BUILTINS.getattr', (var(stack_depth - 1), "'__setitem__'")),
                call(var(stack_depth), var(stack_depth), (var(stack_depth - 2),))]
    elif name == 'BINARY_SUBSCR':
        #
        # return [call(var(out), 'BUILTINS.getattr', (var(stack_depth - 1), "'__getitem__'")),
        #        call(var(out), var(out), (var(stack_depth),))]
        # IVY-Specific: :(
        return [call(var(out), 'BUILTINS.getitem', (var(stack_depth - 1), var(stack_depth)))]
    elif name == 'POP_BLOCK':
        return [NOP]
    elif name == 'SETUP_LOOP':
        return [NOP]
    elif name == 'RAISE_VARARGS':
        return [tac(Op.RAISE, gens=(), uses=[var(stack_depth)], fmt='RAISE {uses[0]}')]
    elif name == 'UNPACK_SEQUENCE':
        seq = [var(stack_depth + i) for i in reversed(range(val))]
        return [mulassign(*seq, rhs=var(stack_depth))]
    elif name == 'IMPORT_NAME':
        return [include(var(out), val)]
    elif name == 'IMPORT_FROM':
        return [include(var(out), var(stack_depth), val)]
    elif name == 'BUILD':
        if op == 'SLICE':
            if val == 2:
                args = (var(stack_depth - 1), var(stack_depth))
            else:
                args = (var(stack_depth), var(stack_depth - 1), var(stack_depth - 2))
            return [call(var(out), 'SLICE', args)]
        return [call(var(out), op, tuple(var(i + 1) for i in range(stack_depth - val, stack_depth)))]
    elif name == 'CALL_FUNCTION':
        nargs = val & 0xFF
        nkwargs = 2 * ((val >> 8) & 0xFF)
        total = nargs + nkwargs
        mid = [var(i + 1) for i in range(stack_depth - nkwargs - nargs, stack_depth - nkwargs)]
        mid_kw = [(var(i) + ': ' + var(i + 1))
                  for i in range(stack_depth - nkwargs + 1, stack_depth, 2)]
        return [call(var(out), var(stack_depth - total), tuple(mid), tuple(mid_kw))]
    elif name == "NOP":
        return []
    return [tac(Op.UNSUPPORTED, fmt=f'UNSUPPORTED {name}')]


def choose_category(opname, argval) -> tuple[str, Optional[str]]:
    # I'm not sure that this function is actually helpful. (ELAZAR)
    if opname in ('UNARY_POSITIVE', 'UNARY_NEGATIVE', 'UNARY_INVERT'):
        return 'UNARY_ATTR', UN_TO_OP[opname]
    if opname in ('GET_ITER', 'UNARY_NOT', 'GET_YIELD_FROM_ITER'):
        return 'UNARY_FUNC', UN_TO_OP[opname]

    if opname in ('JUMP_ABSOLUTE', 'JUMP_FORWARD', 'BREAK_LOOP', 'CONTINUE_LOOP'):
        return 'JUMP', None

    if opname in ("NOP",):
        return 'NOP', None

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
    'POWER': '**',
    'MULTIPLY': '*',
    'MATRIX_MULTIPLY': '@',
    'FLOOR_DIVIDE': '//',
    'TRUE_DIVIDE': '/',
    'MODULO': '%',
    'ADD': '+',
    'SUBTRACT': '-',
    'SUBSCR': '[]',
    'LSHIFT': '<<',
    'RSHIFT': '>>',
    'AND': '&',
    'XOR': '^',
    'OR': '|'
}

# TODO. == to __eq__, etc.
CMPOP_TO_OP = {

}

UN_TO_OP = {
    # __abs__ ?
    'UNARY_POSITIVE': '__pos__',
    'UNARY_NEGATIVE': '__neg__',
    'UNARY_NOT': 'not ',
    'UNARY_INVERT': '__invert__',
    'GET_ITER': 'ITER ',
    'GET_YIELD_FROM_ITER': 'YIELD_FROM_ITER '
}

if __name__ == '__main__':
    test()
