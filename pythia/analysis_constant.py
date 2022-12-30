# Data flow analysis and stuff.

from __future__ import annotations

import typing
from typing import TypeVar, Optional, TypeAlias

from pythia import tac
from .analysis_domain import ValueLattice, TOP, BOTTOM, Top, Bottom

T = TypeVar('T')

Constant: TypeAlias = tac.Const | tac.Predefined | Top | Bottom


class ConstLattice(ValueLattice[Constant]):
    """
    Abstract domain for type analysis with lattice operations.
    """

    def name(self) -> str:
        return "Constant"

    def join(self, left: Constant, right: Constant) -> Constant:
        if self.is_bottom(left) or self.is_top(right):
            return right
        if self.is_bottom(right) or self.is_top(left):
            return left
        if left == right:
            return left
        return self.top()

    def meet(self, left: Constant, right: Constant) -> Constant:
        if self.is_top(left) or self.is_bottom(right):
            return left
        if self.is_top(right) or self.is_bottom(left):
            return right
        if left == right:
            return left
        return self.bottom()

    def top(self) -> Top:
        return TOP

    def initial(self, annotations: dict[tac.Var, str]) -> Constant:
        return self.top()

    def is_top(self, elem: Constant) -> bool:
        return isinstance(elem, Top)

    def is_bottom(self, elem: Constant) -> bool:
        return isinstance(elem, Bottom)

    def is_equivalent(self, left: Constant, right: Constant) -> bool:
        return left == right

    def is_less_than(self, left: Constant, right: Constant) -> bool:
        return self.join(left, right) == right

    def copy(self, value: Constant) -> Constant:
        return value

    def bottom(self) -> Constant:
        return BOTTOM

    def call(self, function: Constant, args: list[Constant]) -> Constant:
        match function:
            case tac.Predefined.LIST: return tac.Const(args)
            case tac.Predefined.TUPLE: return tac.Const(tuple(args))
            case tac.Predefined.SLICE: return tac.Const(slice(*args))
        return self.top()

    def unary(self, value: Constant, op: tac.UnOp) -> Constant:
        if self.is_bottom(value):
            return self.bottom()
        if self.is_top(value):
            return self.top()
        assert isinstance(value, tac.Const)
        const: typing.Any = value.value
        match op:
            case tac.UnOp.NEG: return tac.Const(-const)
            case tac.UnOp.NOT: return tac.Const(not const)
            case tac.UnOp.POS: return tac.Const(+const)
            case tac.UnOp.INVERT: return tac.Const(~const)
        return self.top()

    def binary(self, left: Constant, right: Constant, op: str) -> Constant:
        if self.is_bottom(left) or self.is_bottom(right):
            return self.bottom()
        if self.is_top(left) or self.is_top(right):
            return self.top()
        try:
            assert isinstance(left, tac.Const)
            assert isinstance(right, tac.Const)
            res = eval_binary(op, left, right)
            if res is None:
                return self.bottom()
            return res
        except TypeError:
            return self.top()

    def predefined(self, name: tac.Predefined) -> tac.Predefined:
        return name

    def const(self, value: object) -> Constant:
        return tac.Const(value)

    def attribute(self, var: Constant, attr: tac.Var) -> Constant:
        assert isinstance(attr, tac.Var)
        return self.top()

    def subscr(self, array: Constant, index: Constant) -> Constant:
        return self.top()

    def imported(self, modname: str) -> Constant:
        return tac.Const(tac.Module(modname))


def eval_binary(op: str, left: typing.Any, right: typing.Any) -> Optional[tac.Const]:
    match op:
        case '+': return tac.Const(left + right)
        case '-': return tac.Const(left - right)
        case '*': return tac.Const(left * right)
        case '/': return tac.Const(left / right)
        case '%': return tac.Const(left % right)
        case '**': return tac.Const(left ** right)
        case '&': return tac.Const(left & right)
        case '|': return tac.Const(left | right)
        case '^': return tac.Const(left ^ right)
        case '<<': return tac.Const(left << right)
        case '>>': return tac.Const(left >> right)
        case '>': return tac.Const(left > right)
        case '<': return tac.Const(left < right)
        case '>=': return tac.Const(left >= right)
        case '<=': return tac.Const(left <= right)
        case '==': return tac.Const(left == right)
        case '!=': return tac.Const(left != right)
        case 'in': return tac.Const(left in right)
        case 'is': return tac.Const(left is right)
        case _: raise ValueError(f'unknown binary operator: {op}')
