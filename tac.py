
from __future__ import annotations

import itertools as it
from dataclasses import dataclass
from typing import Iterable, Optional, TypeAlias

import bcode
import bcode_cfg
import graph_utils as gu


def test() -> None:
    import code_examples
    cfg = gu.simplify_cfg(make_tacblock_cfg(code_examples.calc_mandelbrot_vals))
    print_3addr(cfg)
    cfg.draw()


def linearize_cfg(cfg, no_dels=True) -> Iterable[str]:
    for label in sorted(cfg.nodes):
        # The call for sorted() gives us for free the ordering of blocks by "fallthrough"
        # It is not guaranteed anywhere, but it makes sense - the order of blocks is the 
        # order of the lines in the code
        for ins in cfg[label]:
            if no_dels and isinstance(ins, Del):
                continue
            yield f'{label}:\t{ins}'


def print_3addr(cfg, no_dels=True) -> None:
    print('\n'.join(linearize_cfg(cfg, no_dels)))


def make_tacblock_cfg(f) -> gu.Cfg[Tac]:
    depths, cfg = bcode_cfg.make_bcode_block_cfg_from_function(f)

    def bcode_block_to_tac_block(n, block: gu.Block[bcode.BCode]) -> gu.Block[Tac]:
        return gu.ForwardBlock(list(it.chain.from_iterable(
            make_TAC(bc.opname, bc.argval, bc.stack_effect(), depths[bc.offset], bc.starts_line)
            for bc in block)))

    tac_blocks = gu.node_data_map(cfg, bcode_block_to_tac_block)

    # this is simplistic kills analysis, that is not correct in general:
    # tac = [Del(tuple(get_gens(tac) - get_uses(tac))] + tac
    # if stack_effect < 0:
    #     tac += [Del(stackvar(stack_depth + i)) for i in range(stack_effect + 1, 1)]
    return tac_blocks


@dataclass(frozen=True)
class Const:
    value: object

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return repr(self.value)


@dataclass(frozen=True)
class Var:
    name: str
    is_stackvar: bool = False

    def __str__(self):
        if self.is_stackvar:
            return f'${self.name}'
        return self.name

    def __repr__(self):
        return self.__str__()


Value: TypeAlias = Var | Const


@dataclass(frozen=True)
class Attribute:
    var: Var
    attr: str

    def __str__(self):
        return f'{self.var}.{self.attr}'


@dataclass(frozen=True)
class Subscr:
    var: Var
    index: Value

    def __str__(self):
        return f'{self.var}[{self.index}]'


# Simplified version of the real binding construct in Python.
Signature: TypeAlias = Var | tuple[Var] | Attribute | Subscr


@dataclass
class Binary:
    left: Value
    op: str
    right: Value

    def __str__(self):
        return f'{self.left} {self.op} {self.right}'


@dataclass
class Call:
    function: Var
    args: tuple[Value, ...]
    kwargs: Var = None

    def __str__(self):
        res = f'{self.function}{self.args}'
        if self.kwargs:
            res += f', kwargs={self.kwargs}'
        return res


@dataclass
class Yield:
    value: Value


Expr: TypeAlias = Value | Attribute | Subscr | Binary | Call | Yield


def stackvar(x) -> Var:
    return Var(str(x - 1), True)


def is_stackvar(v: Var | Const) -> bool:
    return isinstance(v, Var) and v.is_stackvar


@dataclass
class Nop:
    pass


@dataclass
class Mov:
    """Assignments with no side effect other than setting the value of the LHS."""
    lhs: Var
    rhs: Var | Const

    def __str__(self):
        return f'{self.lhs} = {self.rhs}'


@dataclass
class Assign:
    """Assignments with no control-flow effect (other than exceptions)."""
    lhs: Signature
    expr: Expr

    def __str__(self):
        return f'{self.lhs} = {self.expr}'


@dataclass
class Import:
    lhs: Var
    modname: str
    feature: str = None

    def __str__(self):
        res = f'{self.lhs} = IMPORT {self.modname}'
        if self.feature is not None:
            res += f'.{self.feature}'
        return res


@dataclass
class InplaceBinary:
    lhs: Var
    op: str
    right: Value

    def __str__(self):
        return f'{self.lhs} {self.op}= {self.right}'


@dataclass
class Jump:
    jump_target: str
    cond: Value = True

    def __str__(self):
        return f'IF {self.cond} GOTO {self.jump_target}'


@dataclass
class For:
    lhs: Signature
    iterator: Value
    jump_target: str

    def __str__(self):
        return f'{self.lhs} = next({self.iterator}) HANDLE: GOTO {self.jump_target}'


@dataclass
class Return:
    value: Value

    def __str__(self):
        return f'RETURN {self.value}'


@dataclass
class Raise:
    value: Value

    def __str__(self):
        return f'RAISE {self.value}'


@dataclass
class Del:
    variables: tuple[Var]

    def __str__(self):
        return f'DEL {self.variables}'


@dataclass
class Unsupported:
    name: str


Tac = Nop | Mov | Assign | Import | InplaceBinary | Jump | For | Return | Raise | Del | Unsupported

NOP = Nop()


def free_vars_expr(expr: Expr) -> set[Var]:
    match expr:
        case Const(): return set()
        case Var(): return {expr}
        case Attribute(): return free_vars_expr(expr.var)
        case Subscr(): return free_vars_expr(expr.var)
        case Binary(): return free_vars_expr(expr.left) | free_vars_expr(expr.right)
        case Call(): return {expr.function} | set(it.chain.from_iterable(free_vars_expr(arg) for arg in expr.args))\
                          | ({expr.kwargs} if expr.kwargs else set())
        case Yield(): return free_vars_expr(expr.value)
        case _: raise NotImplementedError(f'free_vars_expr({repr(expr)})')


def free_vars(tac: Tac) -> set[Var]:
    match tac:
        case Nop(): return set()
        case Mov(): return free_vars_expr(tac.rhs)
        case Assign(): return free_vars_expr(tac.expr)
        case Import(): return set()
        case InplaceBinary(): return {tac.lhs, free_vars_expr(tac.right)}
        case Jump(): return free_vars_expr(tac.cond)
        case For(): return free_vars_expr(tac.iterator)
        case Return(): return free_vars_expr(tac.value)
        case Raise(): return free_vars_expr(tac.value)
        case Del(): return set(tac.variables)
        case Unsupported(): return set()
        case _: raise NotImplementedError(f'{tac}')


def gens(tac: Tac) -> set[Var]:
    match tac:
        case Nop(): return set()
        case Mov(): return {tac.lhs}
        case Assign():
            match tac.lhs:
                case Var(): return {tac.lhs}
                case tuple(): return set(tac.lhs)
                case Attribute(): return set()
                case Subscr(): return set()
                case _: raise NotImplementedError(f'gens({tac})')
        case Import(): return {tac.lhs}
        case InplaceBinary(): return {tac.lhs}
        case Jump(): return set()
        case For(): return {tac.lhs}
        case Return(): return set()
        case Raise(): return set()
        case Del(): return set(tac.variables)
        case Unsupported(): return set()
        case _: raise NotImplementedError(f'gens({tac})')


def make_TAC(opname, val, stack_effect, stack_depth, starts_line=None) -> list[Tac]:
    if opname == 'LOAD_CONST' and isinstance(val, tuple):
        lst = []
        for v in val:
            lst += make_TAC('LOAD_CONST', v, 1, stack_depth, starts_line)
            stack_depth += 1
        return lst + make_TAC('BUILD_TUPLE', len(val), -len(val) + 1, stack_depth, starts_line)
    tac = make_TAC_no_dels(opname, val, stack_effect, stack_depth)
    return tac  # [t._replace(starts_line=starts_line) for t in tac]


def make_global(field: str):
    return Attribute(Var('GLOBALS'), field)


def make_nonlocal(field: str):
    return Attribute(Var('NONLOCAL'), field)


def make_TAC_no_dels(opname, val, stack_effect, stack_depth) -> list[Tac]:
    """Translate a bytecode operation into a list of TAC instructions.
    """
    out = stack_depth + stack_effect if stack_depth is not None else None
    name, op = choose_category(opname, val)
    match name:
        case 'UNARY_ATTR':
            return [Assign(stackvar(stack_depth), Call(Var(op), (stackvar(stack_depth),)))]
        case 'UNARY_FUNC':
            return [Assign(stackvar(out), Call(Var(op), (stackvar(stack_depth),)))]
        case 'BINARY':
            lhs = stackvar(out)
            left = stackvar(stack_depth - 1)
            right = stackvar(stack_depth)
            return [Assign(lhs, Binary(left, op, right))]
        case 'INPLACE':
            lhs = stackvar(out)
            right = stackvar(stack_depth)
            return [InplaceBinary(lhs, op, right)]
        case 'POP_JUMP_IF_FALSE' | 'POP_JUMP_IF_TRUE' | 'POP_JUMP_IF_NONE' | 'POP_JUMP_IF_NOT_NONE':
            res: list[Tac] = [Assign(stackvar(stack_depth),
                                     Call(Var('not'), (stackvar(stack_depth),)))] if name.endswith('FALSE') else []
            return res + [Jump(val, stackvar(stack_depth))]
        case 'JUMP':
            return [Jump(val)]
        case 'POP_TOP':
            return []
        case 'DELETE_FAST':
            return []
        case 'ROT_TWO':
            fresh = stackvar(stack_depth + 1)
            return [Mov(fresh, stackvar(stack_depth)),
                    Mov(stackvar(stack_depth), stackvar(stack_depth - 1)),
                    Mov(stackvar(stack_depth - 1), fresh),
                    Del((fresh,))]
        case 'ROT_THREE':
            fresh = stackvar(stack_depth + 1)
            return [Mov(fresh, stackvar(stack_depth - 2)),
                    Mov(stackvar(stack_depth - 2), stackvar(stack_depth - 1)),
                    Mov(stackvar(stack_depth - 1), stackvar(stack_depth)),
                    Mov(stackvar(stack_depth), fresh),
                    Del((fresh,))]
        case 'DUP_TOP':
            return [Mov(stackvar(out), stackvar(stack_depth))]
        case 'DUP_TOP_TWO':
            return [Mov(stackvar(out), stackvar(stack_depth - 2)),
                    Mov(stackvar(out + 1), stackvar(stack_depth - 1))]
        case 'RETURN_VALUE':
            return [Return(stackvar(stack_depth))]
        case 'YIELD_VALUE':
            return [Assign(stackvar(out), Yield(stackvar(stack_depth)))]
        case 'FOR_ITER':
            return [For(stackvar(out), stackvar(stack_depth), val)]
        case 'LOAD':
            lhs = stackvar(out)
            match op:
                case 'ATTR':
                    return [Assign(lhs, Attribute(stackvar(stack_depth), val))]
                case 'METHOD':
                    return [Assign(lhs, Attribute(stackvar(stack_depth), val))]
                case 'FAST' | 'NAME':
                    return [Mov(lhs, Var(val))]
                case 'CONST':
                    return [Mov(lhs, Const(val))]
                case 'DEREF':
                    return [Assign(lhs, make_nonlocal(val))]
                case 'GLOBAL':
                    return [Assign(lhs, make_global(val))]
                case _:
                    assert False, op
        case 'STORE_FAST' | 'STORE_NAME':
            return [Mov(Var(val), stackvar(stack_depth))]
        case 'STORE_GLOBAL':
            return [Assign(make_global(val), stackvar(stack_depth))]
        case 'STORE_ATTR':
            return [Assign(Attribute(stackvar(stack_depth), val), stackvar(stack_depth - 1))]
        case 'STORE_SUBSCR':
            return [Assign(Subscr(stackvar(stack_depth), stackvar(stack_depth - 1)), stackvar(stack_depth - 2))]
        case 'BINARY_SUBSCR':
            #
            # return [call(stackvar(out), 'BUILTINS.getattr', (stackvar(stack_depth - 1), "'__getitem__'")),
            #        call(stackvar(out), stackvar(out), (stackvar(stack_depth),))]
            # IVY-Specific: :(
            return [Assign(stackvar(out), Call(Var('BUILTINS.getitem'), (stackvar(stack_depth - 1), stackvar(stack_depth))))]
        case 'POP_BLOCK':
            return [NOP]
        case 'SETUP_LOOP':
            return [NOP]
        case 'RAISE_VARARGS':
            return [Raise(stackvar(stack_depth))]
        case 'UNPACK_SEQUENCE':
            seq = tuple(stackvar(stack_depth + i) for i in reversed(range(val)))
            return [Assign(seq, stackvar(stack_depth))]
        case 'IMPORT_NAME':
            return [Import(stackvar(out), val)]
        case 'IMPORT_FROM':
            return [Import(stackvar(out), stackvar(stack_depth), val)]
        case 'BUILD':
            if op == 'SLICE':
                if val == 2:
                    args = (stackvar(stack_depth - 1), stackvar(stack_depth))
                else:
                    args = (stackvar(stack_depth), stackvar(stack_depth - 1), stackvar(stack_depth - 2))
                return [Assign(stackvar(out), Call(Var('SLICE'), args))]
            return [Assign(stackvar(out),
                           Call(Var(op), tuple(stackvar(i + 1) for i in range(stack_depth - val, stack_depth))))]
        case 'CALL_FUNCTION':
            nargs = val & 0xFF
            mid = [stackvar(i + 1) for i in range(stack_depth - nargs, stack_depth)]
            return [Assign(stackvar(out), Call(stackvar(stack_depth - nargs), tuple(mid)))]
        case 'CALL_FUNCTION_KW':
            nargs = val
            mid = [stackvar(i + 1) for i in range(stack_depth - nargs - 1, stack_depth - 1)]
            res = [Assign(stackvar(out), Call(stackvar(stack_depth - nargs - 1), tuple(mid), stackvar(stack_depth)))]
            return res
        case "NOP":
            return []
    return [Unsupported(name)]


def choose_category(opname, argval) -> tuple[str, Optional[str]]:
    # NB: I'm not sure that this function is actually helpful.
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
