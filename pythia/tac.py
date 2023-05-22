
from __future__ import annotations as _

import enum
import itertools as it
import sys
import typing
from dataclasses import dataclass, replace
from typing import Optional, TypeAlias

from pythia import disassemble
from pythia import graph_utils as gu
from pythia import instruction_cfg
from pythia.graph_utils import Label, Location


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
    CONST_KEY_MAP = 6
    NOT = 7
    STRING = 8

    def __str__(self) -> str:
        return self.name

    @classmethod
    def lookup(cls, op: str) -> Predefined:
        return cls.__members__[op]


class UnOp(enum.Enum):
    NOT = 0
    POS = 1
    INVERT = 2
    NEG = 3
    ITER = 4
    YIELD_ITER = 5
    NEXT = 6


@dataclass(frozen=True)
class Const:
    value: object

    def __str__(self) -> str:
        return repr(self.value)

    def __repr__(self) -> str:
        return repr(self.value)


@dataclass(frozen=True)
class Var:
    name: str
    is_stackvar: bool = False

    def __str__(self) -> str:
        if self.is_stackvar:
            return f'${self.name}'
        return self.name

    def __repr__(self) -> str:
        return str(self)


Value: TypeAlias = Var | Const
Name: TypeAlias = Var | Predefined


@dataclass(frozen=False)
class Attribute:
    var: Name
    field: Var

    def __str__(self) -> str:
        return f'{self.var}.{self.field}'


@dataclass(frozen=True)
class Subscript:
    var: Var
    index: Var

    def __str__(self) -> str:
        return f'{self.var}[{self.index}]'


# Simplified version of the real binding construct in Python.
Signature: TypeAlias = Var | tuple[Var, ...] | Subscript | Attribute | None


@dataclass(frozen=False)
class Binary:
    left: Var
    op: str
    right: Var
    inplace: bool

    def __str__(self) -> str:
        res = f'{self.left} {self.op} {self.right}'
        return res


@dataclass(frozen=False)
class Unary:
    op: UnOp
    var: Var

    def __str__(self) -> str:
        return f'{self.op.name} {self.var}'


@dataclass(frozen=False)
class Call:
    function: Var | Predefined
    args: tuple[Var, ...]
    kwargs: Optional[Var] = None

    def location(self) -> int:
        return id(self)

    def __str__(self) -> str:
        res = ''
        if self.function != Var('TUPLE'):
            res += f'{self.function}'
        res += f'({", ".join(str(x) for x in self.args)})'
        if self.kwargs:
            res += f', kwargs={self.kwargs}'
        return res


@dataclass
class Yield:
    value: Var


@dataclass
class Import:
    modname: Var | Attribute
    feature: Optional[str] = None

    def __str__(self) -> str:
        res = f'IMPORT {self.modname}'
        if self.feature is not None:
            res += f'.{self.feature}'
        return res


@dataclass
class MakeFunction:
    code: Var
    free_vars: frozenset[Var] = frozenset()
    name: Optional[Var] = None
    annotations: Optional[Var] = None
    defaults: Optional[Var] = None
    kwdefaults: Optional[Var] = None
    positional_only_defaults: Optional[Var] = None
    free_var_cells: Optional[Var] = None

@dataclass
class MakeClass:
    name: str


Expr: TypeAlias = Value | Predefined | Attribute | Subscript | Binary | Unary | Call | Yield | Import | MakeFunction | MakeClass


def stackvar(x: int) -> Var:
    return Var(str(x - 1), True)


def is_stackvar(v: Var | Const) -> bool:
    return isinstance(v, Var) and v.is_stackvar


@dataclass(frozen=True)
class Nop:
    pass


@dataclass(frozen=True)
class Assign:
    """Assignments with no control-flow effect (other than exceptions)."""
    lhs: Optional[Signature]
    expr: Expr

    def __str__(self) -> str:
        if self.lhs is None:
            return f'{self.expr}'
        return f'{self.lhs} = {self.expr}'


@dataclass(frozen=True)
class Jump:
    jump_target: int
    cond: Value = Const(True)

    def __str__(self) -> str:
        if self.cond == Const(True):
            return f'GOTO {self.jump_target}'
        return f'IF {self.cond} GOTO {self.jump_target}'


@dataclass(frozen=True)
class For:
    lhs: Signature
    iterator: Var
    jump_target: int

    def __str__(self) -> str:
        return f'{self.lhs} = next({self.iterator}) HANDLE: GOTO {self.jump_target}'

    def as_call(self) -> Assign:
        return Assign(self.lhs, Unary(UnOp.NEXT, self.iterator))


@dataclass(frozen=True)
class Return:
    value: Var

    def __str__(self) -> str:
        return f'RETURN {self.value}'


@dataclass(frozen=True)
class Raise:
    value: Var

    def __str__(self) -> str:
        return f'RAISE {self.value}'


@dataclass(frozen=True)
class Del:
    variables: tuple[Var]

    def __str__(self) -> str:
        return f'DEL {", ".join(str(x) for x in self.variables)}'


@dataclass(frozen=True)
class Unsupported:
    name: str


Tac = Nop | Assign | Jump | For | Return | Raise | Del | Unsupported

NOP = Nop()


def free_vars_expr(expr: Expr) -> set[Var]:
    match expr:
        case Const(): return set()
        case Var(): return {expr}
        case Attribute(): return {expr.var} if isinstance(expr.var, Var) else set()
        case Subscript(): return free_vars_expr(expr.var) | free_vars_expr(expr.index)
        case Binary(): return free_vars_expr(expr.left) | free_vars_expr(expr.right)
        case Unary(): return free_vars_expr(expr.var)
        case Call():
            return free_vars_expr(expr.function) \
                   | set(it.chain.from_iterable(free_vars_expr(arg) for arg in expr.args)) \
                   | ({expr.kwargs} if expr.kwargs else set())
        case Yield(): return free_vars_expr(expr.value)
        case Import(): return set()
        case MakeFunction(): return set()  # TODO: fix this
        case Predefined(): return set()
        case _: raise NotImplementedError(f'free_vars_expr({repr(expr)})')


def free_vars_lval(signature: Signature) -> set[Var]:
    match signature:
        case Var(): return set()
        case Attribute(): return {signature.var} if isinstance(signature.var, Var) else set()
        case Subscript(): return free_vars_expr(signature.var) | free_vars_expr(signature.index)
        case tuple(): return set(it.chain.from_iterable(free_vars_lval(arg) for arg in signature))
        case None: return set()
        case _: raise NotImplementedError(f'free_vars_lval({repr(signature)})')


def free_vars(tac: Tac) -> set[Var]:
    match tac:
        case Nop(): return set()
        case Assign(): return free_vars_lval(tac.lhs) | free_vars_expr(tac.expr)
        case Jump(): return free_vars_expr(tac.cond)
        case For(): return free_vars_expr(tac.iterator)
        case Return(): return free_vars_expr(tac.value)
        case Raise(): return free_vars_expr(tac.value)
        case Del(): return set(tac.variables)
        case Unsupported(): return set()
        case Predefined(): return set()
        case _: raise NotImplementedError(f'{tac}')


def gens_signature(signature: Signature) -> set[Var]:
    match signature:
        case Var() as lhs: return {lhs}
        case tuple() as items: return set(items)
        case Attribute(): return set()
        case Subscript(): return set()
        case None: return set()
        case _: raise NotImplementedError(f'gens_signature({repr(signature)})')


def gens(tac: Tac) -> set[Var]:
    match tac:
        case Nop(): return set()
        case Assign(lhs=lhs): return gens_signature(lhs)
        case Jump(): return set()
        case For(lhs=lhs): return gens_signature(lhs)
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
                expr = replace(expr, name=new_var)
            if expr.code == target:
                expr = replace(expr, code=new_var)
            return expr
        case Attribute():
            if expr.var == target:
                return replace(expr, var=new_var)
            return expr
        case Call():
            args = tuple(subst_var_in_expr(arg, target, new_var) for arg in expr.args)
            function = new_var if expr.function == target else expr.function
            return replace(expr, function=function, args=args)
        case Subscript():
            if expr.var == target:
                expr = replace(expr, var=new_var)
            if expr.index == target:
                expr = replace(expr, index=new_var)
            return expr
        case Binary():
            if expr.left == target:
                expr = replace(expr, left=new_var)
            if expr.right == target:
                expr = replace(expr, right=new_var)
            return expr
        case Import():
            assert False
        case Yield():
            return replace(expr, value=new_var)
        case Const():
            return expr
        case _:
            raise NotImplementedError(f'subst_var_in_expr({expr}, {target}, {new_var})')


def make_tac(ins: instruction_cfg.Instruction, stack_depth: int,
             trace_origin: dict[int, instruction_cfg.Instruction]) -> list[Tac]:
    stack_effect = instruction_cfg.calculate_stack_effect(ins)
    if ins.opname == 'LOAD_CONST' and isinstance(ins.argval, tuple):
        # We want to handle list and tuple literal in the same way,
        # So we load tuple as if it was a list
        lst = []
        for v in ins.argval:
            lst += make_tac_no_dels('LOAD_CONST', v, 1, stack_depth, ins.argrepr)
            stack_depth += 1
        tac_list = lst + make_tac_no_dels('BUILD_TUPLE', len(ins.argval), -len(ins.argval) + 1, stack_depth, ins.argrepr)
    else:
        tac_list = make_tac_no_dels(ins.opname, ins.argval, stack_effect, stack_depth, ins.argrepr)
    for tac in tac_list:
        trace_origin[id(tac)] = ins
    return tac_list  # [t._replace(starts_line=starts_line) for t in tac]


def make_global(field: str) -> Attribute:
    assert isinstance(field, str)
    return Attribute(Predefined.GLOBALS, Var(field))


def make_nonlocal(field: str) -> Attribute:
    assert isinstance(field, str)
    return Attribute(Predefined.NONLOCALS, Var(field))


def make_class(name: str) -> Attribute:
    assert isinstance(name, str)
    return Attribute(Predefined.NONLOCALS, Var(name))


def make_tac_cfg(f: typing.Any) -> gu.Cfg[Tac]:
    assert sys.version_info[:2] == (3, 11), f'Python version is {sys.version_info} but only 3.11 is supported'
    depths, ins_cfg = instruction_cfg.make_instruction_block_cfg_from_function(f)

    # simplified_cfg = gu.simplify_cfg(ins_cfg)
    # gu.pretty_print_cfg(simplified_cfg)

    trace_origin: dict[int, instruction_cfg.Instruction] = {}

    def instruction_block_to_tac_block(n: Label, block: gu.Block[instruction_cfg.Instruction]) -> gu.Block[Tac]:
        return gu.Block(list(it.chain.from_iterable(make_tac(ins, depths[ins.offset], trace_origin)
                                                           for ins in block)))

    def annotator(location: Location, n: Tac) -> str:
        pos = trace_origin[id(n)].positions
        if pos is None:
            return f'None'
        return f'{pos.lineno}:{pos.col_offset}'

    tac_cfg: gu.Cfg[Tac] = gu.node_data_map(ins_cfg, instruction_block_to_tac_block)
    tac_cfg.annotator = annotator
    return tac_cfg


def make_tac_no_dels(opname: str, val: str | int | None, stack_effect: int, stack_depth: int, argrepr: str) -> list[Tac]:
    """Translate a bytecode operation into a list of TAC instructions.
    """
    out = stack_depth + stack_effect if stack_depth is not None else None
    match opname.split('_'):
        case ['UNARY', sop] | ['GET', 'ITER' as sop] | ['GET', 'YIELD' as sop, 'FROM', 'ITER']:
            match sop:
                case 'POSITIVE': op = UnOp.POS
                case 'NEGATIVE': op = UnOp.NEG
                case 'INVERT': op = UnOp.INVERT
                case 'NOT': op = UnOp.NOT
                case 'ITER': op = UnOp.ITER
                case 'YIELD': op = UnOp.YIELD_ITER
                case _: raise NotImplementedError(f'UNARY_{sop}')
            return [Assign(stackvar(stack_depth), Unary(op, stackvar(stack_depth)))]
        case ['BINARY', 'SUBSCR']:
            #
            # return [call(stackvar(out), 'BUILTINS.getattr', (stackvar(stack_depth - 1), "'__getitem__'")),
            #        call(stackvar(out), stackvar(out), (stackvar(stack_depth),))]
            # IVY-Specific: :(
            return [Assign(stackvar(out), Subscript(stackvar(stack_depth - 1), stackvar(stack_depth)))]
        case ['BINARY' | 'COMPARE', 'OP']:
            lhs = stackvar(out)
            left = stackvar(stack_depth - 1)
            right = stackvar(stack_depth)
            if argrepr[-1] == '=' and argrepr[0] != '=':
                return [Assign(lhs, Binary(left, argrepr[:-1], right, inplace=True))]
            else:
                return [Assign(lhs, Binary(left, argrepr, right, inplace=False))]
        case ['POP', 'JUMP', 'FORWARD' | 'BACKWARD', 'IF', *v]:
            sop = '_'.join(v)
            # 'FALSE' | 'TRUE' | 'NONE' | 'NOT_NONE'
            res: list[Tac]
            if sop == 'FALSE':
                res = [Assign(stackvar(stack_depth), Unary(UnOp.NOT, stackvar(stack_depth)))]
            else:
                res = []
            assert isinstance(val, int)
            return res + [Jump(val, stackvar(stack_depth))]
        case ['JUMP', 'ABSOLUTE' | 'FORWARD' | 'BACKWARD'] | ['BREAK', 'LOOP'] | ['CONTINUE', 'LOOP']:
            assert isinstance(val, int)
            return [Jump(val)]
        case ['POP', 'TOP']:
            return []
        case ['DELETE', 'FAST']:
            variables = (Var(argrepr, False),)
            return [Del(variables)]
        case ['ROT', 'TWO']:
            fresh = stackvar(stack_depth + 1)
            return [Assign(fresh, stackvar(stack_depth)),
                    Assign(stackvar(stack_depth), stackvar(stack_depth - 1)),
                    Assign(stackvar(stack_depth - 1), fresh),
                    Del((fresh,))]
        case ['ROT', 'THREE']:
            fresh = stackvar(stack_depth + 1)
            return [Assign(fresh, stackvar(stack_depth - 2)),
                    Assign(stackvar(stack_depth - 2), stackvar(stack_depth - 1)),
                    Assign(stackvar(stack_depth - 1), stackvar(stack_depth)),
                    Assign(stackvar(stack_depth), fresh),
                    Del((fresh,))]
        case ['DUP', 'TOP']:
            return [Assign(stackvar(out), stackvar(stack_depth))]
        case ['DUP', 'TOP', 'TWO']:
            return [Assign(stackvar(out), stackvar(stack_depth - 2)),
                    Assign(stackvar(out + 1), stackvar(stack_depth - 1))]
        case ['RETURN', 'VALUE']:
            return [Return(stackvar(stack_depth))]
        case ['YIELD', 'VALUE']:
            return [Assign(stackvar(out), Yield(stackvar(stack_depth)))]
        case ['FOR', 'ITER']:
            assert isinstance(val, int)
            return [For(stackvar(out), stackvar(stack_depth), val)]
        case ['LOAD', 'CONST']:
            return [Assign(stackvar(out), Const(val))]
        case ['LOAD', *ops]:
            lhs = stackvar(out)
            assert isinstance(val, str), f'{opname}, {val}, {argrepr}'
            match ops:
                case ['ATTR']:
                    return [Assign(lhs, Attribute(stackvar(stack_depth), Var(val)))]
                case ['METHOD']:
                    return [Assign(lhs, Attribute(stackvar(stack_depth), Var(val)))]
                case ['FAST' | 'NAME']:
                    return [Assign(lhs, Var(val))]
                case ['DEREF']:
                    return [Assign(lhs, make_nonlocal(val))]
                case ['GLOBAL']:
                    return [Assign(lhs, make_global(val))]
                case ['CLOSURE']:
                    print("Unknown: LOAD CLOSURE")
                    return [Assign(lhs, Const(None))]
                case ['BUILD', 'CLASS']:
                    return [Assign(lhs, make_class(val))]
                case ['ASSERTION', 'ERROR']:
                    return [Assign(lhs, Const(AssertionError()))]
                case _:
                    assert False, ops
        case ['STORE', 'FAST' | 'NAME']:
            assert isinstance(val, str)
            return [Assign(Var(val), stackvar(stack_depth))]
        case ['STORE', 'GLOBAL']:
            assert isinstance(val, str)
            return [Assign(make_global(val), stackvar(stack_depth))]
        case ['STORE', 'ATTR']:
            assert isinstance(val, str)
            attr = Attribute(stackvar(stack_depth), Var(val))
            return [Assign(attr, stackvar(stack_depth - 1))]
        case ['STORE', 'SUBSCR']:
            return [Assign(Subscript(stackvar(stack_depth - 1), stackvar(stack_depth)), stackvar(stack_depth - 2))]
        case ['POP', 'BLOCK']:
            return []
        case ['SETUP', 'LOOP']:
            return []
        case ['RAISE', 'VARARGS']:
            return [Raise(stackvar(stack_depth))]
        case ['UNPACK', 'SEQUENCE']:
            assert isinstance(val, int)
            seq = tuple(stackvar(stack_depth + i) for i in reversed(range(val)))
            return [Assign(seq, stackvar(stack_depth))]
        case ['IMPORT', 'NAME']:
            assert isinstance(val, str)
            return [Assign(stackvar(out), Import(Var(val)))]
        case ['IMPORT', 'FROM']:
            assert isinstance(val, str)
            return [Assign(stackvar(out), Import(Attribute(stackvar(stack_depth), Var(val))))]
        case ['BUILD', 'SLICE']:
            args: tuple[Var, ...]
            if val == 2:
                args = (stackvar(stack_depth - 1), stackvar(stack_depth))
            else:
                args = (stackvar(stack_depth), stackvar(stack_depth - 1), stackvar(stack_depth - 2))
            return [Assign(stackvar(out), Call(Predefined.SLICE, args))]
        case ['BUILD', op]:
            assert isinstance(val, int)
            return [Assign(stackvar(out),
                           Call(Predefined.lookup(op), tuple(stackvar(i + 1) for i in range(stack_depth - val, stack_depth))))]
        case ['SWAP']:
            a = stackvar(stack_depth - 0)
            b = stackvar(stack_depth - 1)
            return [Assign(a, Call(Predefined.TUPLE, (b, a))),
                    Assign((a, b), a)]
        case ['CALL']:
            assert isinstance(val, int)
            nargs = val & 0xFF
            mid = [stackvar(i + 1) for i in range(stack_depth, stack_depth + nargs)]
            return [Assign(stackvar(out), Call(stackvar(stack_depth), tuple(mid)))]
        case ['CALL', 'FUNCTION', 'KW']:
            assert isinstance(val, int)
            nargs = val
            mid = [stackvar(i + 1) for i in range(stack_depth - nargs - 1, stack_depth - 1)]
            res = [Assign(stackvar(out), Call(stackvar(stack_depth - nargs - 1), tuple(mid), stackvar(stack_depth)))]
            return res
        case ["NOP" | 'RESUME' | 'PRECALL']:
            return []
        case ['EXTENDED', 'ARG']:
            """
            Prefixes any opcode which has an argument too big to fit into the default one byte.
            ext holds an additional byte which act as higher bits in the argument.
            For each opcode, at most three prefixal EXTENDED_ARG are allowed,
            forming an argument from two-byte to four-byte.
            """
            return []
        case ['KW', 'NAMES']:
            """
            Prefixes PRECALL. Stores a reference to co_consts[consti] into an internal variable for use by CALL.
            co_consts[consti] must be a tuple of strings.
            """
            return []
        case ["MAKE", "FUNCTION"]:
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
            function = MakeFunction(stackvar(stack_depth))
            assert isinstance(val, int)
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
    return [Unsupported(opname)]


def main() -> None:
    env, imports = disassemble.read_file('examples/toy.py')

    for k, func in env.items():
        cfg = make_tac_cfg(func)
        cfg = gu.simplify_cfg(cfg)
        gu.pretty_print_cfg(cfg)
