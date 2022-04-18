# Data flow analysis and stuff.

from __future__ import annotations

from dataclasses import dataclass
from typing import Type, TypeVar, Optional, ClassVar

import tac
from tac import Const, Var
from tac_analysis_domain import AbstractDomain, IterationStrategy, ForwardIterationStrategy, Bottom

import graph_utils as gu

T = TypeVar('T')


class ConstantDomain(AbstractDomain):
    constants: dict[Var, Const] | Bottom

    BOTTOM: ClassVar[Bottom] = Bottom()

    @staticmethod
    def name() -> str:
        return "Constant"

    @classmethod
    def view(cls, cfg: gu.Cfg[T]) -> IterationStrategy[T]:
        return ForwardIterationStrategy(cfg)

    def __init__(self, constants: dict[Var, Const] | Bottom) -> None:
        super().__init__()
        if constants is ConstantDomain.BOTTOM:
            self.constants = ConstantDomain.BOTTOM
        else:
            self.constants = constants.copy() or {}

    def __le__(self, other):
        return self.join(other).constants == other.constants

    def __eq__(self, other):
        return self.constants == other.constants

    def copy(self: T) -> T:
        return ConstantDomain(self.constants)

    @classmethod
    def initial(cls: Type[T]) -> T:
        return cls.top()

    @classmethod
    def top(cls: Type[T]) -> T:
        return ConstantDomain({})

    @classmethod
    def bottom(cls: Type[T]) -> T:
        return ConstantDomain(ConstantDomain.BOTTOM)

    @property
    def is_bottom(self) -> bool:
        return self.constants is ConstantDomain.BOTTOM

    def join(self: T, other: T) -> T:
        if self.is_bottom:
            return other.copy()
        if other.is_bottom:
            return self.copy()
        return ConstantDomain(dict(self.constants.items() & other.constants.items()))

    def transfer(self, ins: tac.Tac, location: str) -> None:
        if self.is_bottom:
            return
        constants = self.constants.copy()
        for var in tac.gens(ins):
            if var in self.constants:
                del self.constants[var]
        if isinstance(ins, tac.Mov):
            if isinstance(ins.rhs, tac.Const):
                self.constants[ins.lhs] = ins.rhs
            elif ins.rhs in self.constants:  # and ins.target.is_stackvar:
                self.constants[ins.lhs] = constants[ins.rhs]
        elif isinstance(ins, tac.Assign):
            if isinstance(ins.lhs, tac.Var) and (val := eval(constants, ins.expr)) is not None:
                self.constants[ins.lhs] = val
        elif isinstance(ins, tac.Import):
            self.constants[ins.lhs] = tac.Const(tac.Module(ins.modname))

    def __str__(self) -> str:
        return 'Constants({})'.format(", ".join(f'{k}={v}' for k, v in self.constants.items()))

    def __repr__(self) -> str:
        return self.constants.__repr__()

    def keep_only_live_vars(self, alive_vars: set[tac.Var]) -> None:
        for var in set(self.constants.keys()) - alive_vars:
            del self.constants[var]


def eval(constants: dict[Var, Const], expr: tac.Expr) -> Optional[Const]:
    match expr:
        case tac.Const(): return expr
        case tac.Var(): return constants.get(expr)
        case tac.Attribute(): return None
        case tac.Call(): return None
        case tac.Subscr(): return None
        case tac.Yield(): return None
        case tac.Binary():
            left = eval(constants, expr.left)
            right = eval(constants, expr.right)
            if left is not None and right is not None:
                try:
                    return eval_binary(expr.op, left.value, right.value)
                except ValueError:
                    return None
            else:
                return None


def eval_binary(op: str, left: object, right: object) -> Optional[Const]:
    match op:
        case '+': return Const(left + right)
        case '-': return Const(left - right)
        case '*': return Const(left * right)
        case '/': return Const(left / right)
        case '%': return Const(left % right)
        case '**': return Const(left ** right)
        case '&': return Const(left & right)
        case '|': return Const(left | right)
        case '^': return Const(left ^ right)
        case '<<': return Const(left << right)
        case '>>': return Const(left >> right)
        case '>': return Const(left > right)
        case '<': return Const(left < right)
        case '>=': return Const(left >= right)
        case '<=': return Const(left <= right)
        case '==': return Const(left == right)
        case '!=': return Const(left != right)
        case 'in': return Const(left in right)
        case 'is': return Const(left is right)
        case _: raise ValueError(f'unknown binary operator: {op}')
