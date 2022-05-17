# Data flow analysis and stuff.

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, Optional

import tac
from tac import Const
from tac_analysis_domain import Lattice, TOP, BOTTOM, Top, Bottom

T = TypeVar('T')


@dataclass(frozen=True)
class ConstLattice(Lattice[Const]):
    """
    Abstract domain for type analysis with lattice operations.
    """

    @staticmethod
    def name() -> str:
        return "Const"

    def join(self, left: Const, right: Const) -> Const:
        if self.is_bottom(left) or self.is_top(right):
            return right
        if self.is_bottom(right) or self.is_top(left):
            return left
        if left == right:
            return left
        return self.top()

    def meet(self, left: Const, right: Const) -> Const:
        if self.is_top(left) or self.is_bottom(right):
            return left
        if self.is_top(right) or self.is_bottom(left):
            return right
        if left == right:
            return left
        return self.bottom()

    def top(self) -> Const:
        return TOP

    def is_top(self, elem: Const) -> bool:
        return isinstance(elem, Top)

    def is_bottom(self, elem: Const) -> bool:
        return isinstance(elem, Bottom)

    @classmethod
    def bottom(cls) -> Const:
        return BOTTOM

    def call(self, function: Const, args: list[Const]) -> Const:
        return self.top()

    def binary(self, left: Const, right: Const, op: str) -> Const:
        if self.is_bottom(left) or self.is_bottom(right):
            return self.bottom()
        if self.is_top(left) or self.is_top(right):
            return self.top()
        try:
            return eval_binary(op, left, right)
        except ValueError:
            return self.top()

    def predefined(self, name: tac.Predefined) -> Optional[Const]:
        return self.top()

    def const(self, value: object) -> Const:
        return Const(value)

    def attribute(self, var: Const, attr: str) -> Const:
        return self.top()

    def subscr(self, array: Const, index: Const) -> Const:
        return self.top()

    def annotation(self, code: str) -> Const:
        return self.top()

    def imported(self, modname: str) -> Const:
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
