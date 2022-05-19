
from __future__ import annotations

import enum
import itertools as it
import dataclasses
from dataclasses import dataclass
from typing import Iterable, Optional, TypeAlias

import bcode
import bcode_cfg
import disassemble
import graph_utils as gu


def test() -> None:
    code = disassemble.read_function('__pycache__/code_examples.cpython-310.pyc', 'simple')
    cfg = gu.simplify_cfg(make_tacblock_cfg(code))
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

    return gu.node_data_map(cfg, bcode_block_to_tac_block)


@dataclass(frozen=True)
class Module:
    name: str

    def __repr__(self) -> str:
        return f'Module({self.name})'


class Predefined(enum.Enum):
    GLOBALS = 0
    LOCALS = 1
    NONLOCALS = 2
    LIST = 3
    TUPLE = 4
    SLICE = 5

    def __str__(self):
        return self.name

    @classmethod
    def lookup(cls, op):
        return cls.__members__[op]


@dataclass(frozen=True)
class Const:
    value: object

    def __str__(self):
        return repr(self.value)

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


Value: TypeAlias = Var | Const
Name: TypeAlias = Var | Predefined


@dataclass(frozen=True)
class Attribute:
    var: Name
    attr: Var

    def __str__(self):
        return f'{self.var}.{self.attr}'


@dataclass(frozen=True)
class Subscript:
    var: Var
    index: Value

    def __str__(self):
        return f'{self.var}[{self.index}]'


# Simplified version of the real binding construct in Python.
Signature: TypeAlias = Var | tuple[Var]


@dataclass
class Binary:
    left: Value
    op: str
    right: Value
    is_allocation: bool = False

    def __str__(self):
        res = f'{self.left} {self.op} {self.right}'
        if self.is_allocation:
            res += ' #  new'
        return res


@dataclass(frozen=False)
class Call:
    function: Var | Attribute
    args: tuple[Value, ...]
    kwargs: Var = None
    is_allocation: bool = False

    def location(self) -> int:
        return id(self)

    def __str__(self):
        res = ''
        if self.function != Var('TUPLE'):
            res += f'{self.function}'
        res += f'({", ".join(str(x) for x in self.args)})'
        if self.kwargs:
            res += f', kwargs={self.kwargs}'
        if self.is_allocation:
            res += ' #  new'
        return res


@dataclass
class Yield:
    value: Value


@dataclass
class Import:
    modname: str
    feature: str = None

    def __str__(self):
        res = f'IMPORT {self.modname}'
        if self.feature is not None:
            res += f'.{self.feature}'
        return res

@dataclass
class MakeFunction:
    name: Value
    code: Value
    free_vars: set[Var] = frozenset()
    annotations: dict[Var, str] = None
    defaults: dict[Var, Var] = None
    positional_only_defaults: dict[Var, Var] = None


Expr: TypeAlias = Value | Attribute | Subscript | Binary | Call | Yield | Import | MakeFunction


def stackvar(x) -> Var:
    return Var(str(x - 1), True)


def is_stackvar(v: Var | Const) -> bool:
    return isinstance(v, Var) and v.is_stackvar


@dataclass
class Nop:
    pass


@dataclass(frozen=True)
class Assign:
    """Assignments with no control-flow effect (other than exceptions)."""
    lhs: Optional[Signature]
    expr: Expr

    def __str__(self):
        if self.lhs is None:
            return f'{self.expr}'
        return f'{self.lhs} = {self.expr}'

    @property
    def is_mov(self) -> bool:
        return isinstance(self.lhs, Var) and isinstance(self.expr, (Var, Attribute, Const))

    @property
    def no_side_effect(self) -> bool:
        return self.assign_stack and isinstance(self.expr, (Subscript, Attribute, Var, Const))

    @property
    def assign_stack(self) -> bool:
        return self.lhs is None or isinstance(self.lhs, Var) and is_stackvar(self.lhs)


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
    cond: Value = Const(True)

    def __str__(self):
        return f'IF {self.cond} GOTO {self.jump_target}'


@dataclass
class For:
    lhs: Signature
    iterator: Value
    jump_target: str

    def __str__(self):
        return f'{self.lhs} = next({self.iterator}) HANDLE: GOTO {self.jump_target}'

    def as_call(self) -> Assign:
        return Assign(self.lhs, Call(Attribute(self.iterator, Var('__next__')), ()))


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


Tac = Nop | Assign | InplaceBinary | Jump | For | Return | Raise | Del | Unsupported

NOP = Nop()


def free_vars_expr(expr: Expr) -> set[Var]:
    match expr:
        case Const(): return set()
        case Var(): return {expr}
        case Attribute(): return {expr.var}
        case Subscript(): return free_vars_expr(expr.var) | free_vars_expr(expr.index)
        case Binary(): return free_vars_expr(expr.left) | free_vars_expr(expr.right)
        case Call():
            return free_vars_expr(expr.function) | set(it.chain.from_iterable(free_vars_expr(arg) for arg in expr.args)) \
                   | ({expr.kwargs} if expr.kwargs else set())
        case Yield(): return free_vars_expr(expr.value)
        case Import(): return set()
        case MakeFunction(): return {expr.name, expr.code}  # TODO: fix this
        case _: raise NotImplementedError(f'free_vars_expr({repr(expr)})')


def free_vars_lval(signature: Signature) -> set[Var]:
    match signature:
        case Var(): return set()
        case Attribute(): return {signature.var}
        case Subscript(): return free_vars_expr(signature.var) | free_vars_expr(signature.index)
        case tuple(): return set(it.chain.from_iterable(free_vars_lval(arg) for arg in signature))
        case None: return set()
        case _: raise NotImplementedError(f'free_vars_lval({repr(signature)})')


def free_vars(tac: Tac) -> set[Var]:
    match tac:
        case Nop(): return set()
        case Assign(): return free_vars_lval(tac.lhs) | free_vars_expr(tac.expr)
        case InplaceBinary(): return {tac.lhs} | free_vars_expr(tac.right)
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
        case Assign():
            match tac.lhs:
                case Var(): return {tac.lhs}
                case tuple(): return set(tac.lhs)
                case Attribute(): return set()
                case Subscript(): return set()
                case None: return set()
                case _: raise NotImplementedError(f'gens({tac})')
        case InplaceBinary(): return {tac.lhs}
        case Jump(): return set()
        case For(): return {tac.lhs}
        case Return(): return set()
        case Raise(): return set()
        case Del(): return set(tac.variables)
        case Unsupported(): return set()
        case _: raise NotImplementedError(f'gens({tac})')


def subst_var_in_expr(expr: Expr, target: Var, new_var: Var) -> Expr:
    match expr:
        case Var():
            return new_var if expr == target else expr
        case MakeFunction():
            if expr.name == target:
                expr = dataclasses.replace(expr, name=new_var)
            if expr.code == target:
                expr = dataclasses.replace(expr, code=new_var)
            return expr
        case Attribute():
            if expr.var == target:
                return dataclasses.replace(expr, var=new_var)
            return expr
        case Call():
            args = tuple(subst_var_in_expr(arg, target, new_var) for arg in expr.args)
            function = new_var if expr.function == target else expr.function
            return dataclasses.replace(expr, function=function, args=args)
        case Subscript():
            if expr.var == target:
                expr = dataclasses.replace(expr, var=new_var)
            if expr.index == target:
                expr = dataclasses.replace(expr, index=new_var)
            return expr
        case Binary():
            if expr.left == target:
                expr = dataclasses.replace(expr, left=new_var)
            if expr.right == target:
                expr = dataclasses.replace(expr, right=new_var)
            return expr
        case Import():
            assert False
        case Yield():
            return dataclasses.replace(expr, value=new_var)
        case Const():
            return expr
        case _:
            raise NotImplementedError(f'subst_var_in_expr({expr}, {target}, {new_var})')


def subst_var_in_signature(signature: Signature, target: Var, new_var: Var) -> Signature:
    match signature:
        case Var():
            return signature
        case tuple():
            return tuple(subst_var_in_signature(arg, target, new_var) for arg in signature)
        case Attribute():
            if signature.var == target:
                return dataclasses.replace(signature, var=new_var)
            return signature
        case Subscript():
            if signature.var == target:
                signature = dataclasses.replace(signature, var=new_var)
            if signature.index == target:
                signature = dataclasses.replace(signature, index=new_var)
            return signature
    return signature


def subst_var_in_ins(ins: Tac, target: Var, new_var: Var) -> Tac:
    match ins:
        case Assign():
            return dataclasses.replace(ins,
                                       lhs=subst_var_in_signature(ins.lhs, target, new_var),
                                       expr=subst_var_in_expr(ins.expr, target, new_var))
        case InplaceBinary():
            return dataclasses.replace(ins,
                                       lhs=subst_var_in_signature(ins.lhs, target, new_var),
                                       right=subst_var_in_expr(ins.right, target, new_var))
        case Return():
            return dataclasses.replace(ins, value=subst_var_in_expr(ins.value, target, new_var))
        case Yield():
            return dataclasses.replace(ins, value=subst_var_in_expr(ins.value, target, new_var))
        case For():
            return dataclasses.replace(ins,
                                       lhs=subst_var_in_signature(ins.lhs, target, new_var),
                                       iterator=subst_var_in_expr(ins.iterator, target, new_var))
        case Raise():
            return dataclasses.replace(ins,
                                       value=subst_var_in_expr(ins.value, target, new_var))
    return ins


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
    return Attribute(Predefined.GLOBALS, Var(field))


def make_nonlocal(field: str):
    return Attribute(Predefined.NONLOCALS, Var(field))


def make_TAC_no_dels(opname, val, stack_effect, stack_depth) -> list[Tac]:
    """Translate a bytecode operation into a list of TAC instructions.
    """
    out = stack_depth + stack_effect if stack_depth is not None else None
    name, op = choose_category(opname, val)
    match name:
        case 'UNARY_ATTR':
            return [Assign(stackvar(stack_depth), Call(Var(op), (stackvar(stack_depth),)))]
        case 'UNARY_FUNC':
            return [Assign(stackvar(out), Call(Attribute(stackvar(stack_depth), Var(op)), ()))]
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
            return [Assign(fresh, stackvar(stack_depth)),
                    Assign(stackvar(stack_depth), stackvar(stack_depth - 1)),
                    Assign(stackvar(stack_depth - 1), fresh),
                    Del((fresh,))]
        case 'ROT_THREE':
            fresh = stackvar(stack_depth + 1)
            return [Assign(fresh, stackvar(stack_depth - 2)),
                    Assign(stackvar(stack_depth - 2), stackvar(stack_depth - 1)),
                    Assign(stackvar(stack_depth - 1), stackvar(stack_depth)),
                    Assign(stackvar(stack_depth), fresh),
                    Del((fresh,))]
        case 'DUP_TOP':
            return [Assign(stackvar(out), stackvar(stack_depth))]
        case 'DUP_TOP_TWO':
            return [Assign(stackvar(out), stackvar(stack_depth - 2)),
                    Assign(stackvar(out + 1), stackvar(stack_depth - 1))]
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
                    return [Assign(lhs, Attribute(stackvar(stack_depth), Var(val)))]
                case 'METHOD':
                    return [Assign(lhs, Attribute(stackvar(stack_depth), Var(val)))]
                case 'FAST' | 'NAME':
                    return [Assign(lhs, Var(val))]
                case 'CONST':
                    return [Assign(lhs, Const(val))]
                case 'DEREF':
                    return [Assign(lhs, make_nonlocal(val))]
                case 'GLOBAL':
                    return [Assign(lhs, make_global(val))]
                case 'CLOSURE':
                    print("Uknown: LOAD CLOSURE")
                    return [Assign(lhs, Const(None))]
                case _:
                    assert False, op
        case 'STORE_FAST' | 'STORE_NAME':
            return [Assign(Var(val), stackvar(stack_depth))]
        case 'STORE_GLOBAL':
            return [Assign(make_global(val), stackvar(stack_depth))]
        case 'STORE_ATTR':
            return [Assign(Attribute(stackvar(stack_depth), Var(val)), stackvar(stack_depth - 1))]
        case 'STORE_SUBSCR':
            return [Assign(Subscript(stackvar(stack_depth - 1), stackvar(stack_depth)), stackvar(stack_depth - 2))]
        case 'BINARY_SUBSCR':
            #
            # return [call(stackvar(out), 'BUILTINS.getattr', (stackvar(stack_depth - 1), "'__getitem__'")),
            #        call(stackvar(out), stackvar(out), (stackvar(stack_depth),))]
            # IVY-Specific: :(
            return [Assign(stackvar(out), Subscript(stackvar(stack_depth - 1), stackvar(stack_depth)))]
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
            return [Assign(stackvar(out), Import(val))]
        case 'IMPORT_FROM':
            return [Assign(stackvar(out), Import(stack_depth, val))]
        case 'BUILD':
            if op == 'SLICE':
                if val == 2:
                    args = (stackvar(stack_depth - 1), stackvar(stack_depth))
                else:
                    args = (stackvar(stack_depth), stackvar(stack_depth - 1), stackvar(stack_depth - 2))
                return [Assign(stackvar(out), Call(Predefined.SLICE, args))]
            return [Assign(stackvar(out),
                           Call(Predefined.lookup(op), tuple(stackvar(i + 1) for i in range(stack_depth - val, stack_depth))))]
        case 'CALL_FUNCTION' | 'CALL_METHOD':
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
        case "MAKE_FUNCTION":
            """
            MAKE_FUNCTION(argc)
            Pushes a new function object on the stack.
            From bottom to top, the consumed stack must consist of values
            if the argument carries a specified flag value:
            
            0x01   a tuple of default values for positional-only and positional-or-keyword parameters in positional order
            0x02   a dictionary of keyword-only parameters’ default values
            0x04   a tuple of strings containing parameters’ annotations
            0x08   a tuple containing cells for free variables, making a closure
            the code associated with the function (at TOS1)
            the qualified name of the function (at TOS)
            """
            function = MakeFunction(stackvar(stack_depth), stackvar(stack_depth - 1))
            i = 2
            if val & 0x01:
                function.defaults = stackvar(stack_depth - i)
                i += 1
            if val & 0x02:
                function.kwdefaults = stackvar(stack_depth - i)
                i += 1
            if val & 0x04:
                function.annotations = stackvar(stack_depth - i)
                i += 1
            if val & 0x08:
                function.free_var_cells = stackvar(stack_depth - i)
            return [Assign(stackvar(out), function)]
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
    'UNARY_NOT': '__bool__',  # not exactly the same semantics
    'UNARY_INVERT': '__invert__',
    'GET_ITER': '__iter__',
    'GET_YIELD_FROM_ITER': 'YIELD_FROM_ITER '
}

if __name__ == '__main__':
    test()
