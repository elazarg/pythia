# Data flow analysis and stuff.

from __future__ import annotations

from typing import TypeVar, Optional, TypeAlias

import tac
from tac import Const, Predefined
from tac_analysis_domain import ActionLattice, TOP, BOTTOM, Top, Bottom

T = TypeVar('T')

Constant: TypeAlias = Const | Predefined | Top | Bottom


class ConstLattice(ActionLattice[Constant]):
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

    def top(self) -> Constant:
        return TOP

    def is_top(self, elem: Constant) -> bool:
        return isinstance(elem, Top)

    def is_bottom(self, elem: Constant) -> bool:
        return isinstance(elem, Bottom)

    @classmethod
    def bottom(cls) -> Constant:
        return BOTTOM

    def call(self, function: Constant, args: list[Constant]) -> Constant:
        match function:
            case Predefined.LIST: return Const(args)
            case Predefined.TUPLE: return Const(tuple(args))
            case Predefined.SLICE: return Const(slice(*args))
        return self.top()

    def unary(self, value: Const, op: tac.UnOp) -> Const:
        if self.is_bottom(value):
            return self.bottom()
        if self.is_top(value):
            return self.top()
        match op:
            case tac.UnOp.NEG: return Const(-value.value)
            case tac.UnOp.NOT: return Const(not value.value)
            case tac.UnOp.POS: return Const(+value.value)
            case tac.UnOp.INVERT: return Const(~value.value)
        return self.top()

    def binary(self, left: Constant, right: Constant, op: str) -> Constant:
        if self.is_bottom(left) or self.is_bottom(right):
            return self.bottom()
        if self.is_top(left) or self.is_top(right):
            return self.top()
        try:
            return eval_binary(op, left, right)
        except TypeError:
            return self.top()

    def predefined(self, name: tac.Predefined) -> Optional[Constant]:
        return name

    def const(self, value: object) -> Constant:
        return Const(value)

    def attribute(self, var: Constant, attr: tac.Var) -> Constant:
        assert isinstance(attr, tac.Var)
        return self.top()

    def subscr(self, array: Constant, index: Constant) -> Constant:
        return self.top()

    def imported(self, modname: str) -> Constant:
        return tac.Const(tac.Module(modname))


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
