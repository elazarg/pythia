# Data flow analysis and stuff.

from __future__ import annotations
from typing import Type, TypeVar
import typing

import tac
from tac import Tac
from tac_analysis_domain import AbstractDomain

T = TypeVar('T')


class ConstantDomain(AbstractDomain):
    @classmethod
    def is_forward(cls) -> bool:
        return True

    def __init__(self, constants: typing.Optional[dict[tac.Var, tac.Const]] = ()) -> None:
        super().__init__()
        if constants is None:
            self.constants = None
        else:
            self.constants = constants or {}

    def __le__(self, other):
        return self.join(other).constants == other.constants

    def __eq__(self, other):
        return self.constants == other.constants

    def __ne__(self, other):
        return self.constants != other.constants

    def copy(self: T) -> T:
        return ConstantDomain(self.constants)

    @classmethod
    def top(cls: Type[T]) -> T:
        return ConstantDomain({})

    def set_to_top(self) -> None:
        self.constants = {}

    @classmethod
    def bottom(cls: Type[T]) -> T:
        return ConstantDomain(None)

    @property
    def is_bottom(self) -> bool:
        return self.constants is None

    def join(self: T, other: T) -> T:
        if self.is_bottom:
            return other.copy()
        if other.is_bottom:
            return self.copy()
        return ConstantDomain(dict(self.constants.items() & other.constants.items()))

    def transfer(self, ins: tac.Tac) -> None:
        if self.is_bottom:
            return
        if isinstance(ins, tac.Mov):
            if isinstance(ins.rhs, tac.Const):
                self.constants[ins.lhs] = ins.rhs
            elif ins.rhs in self.constants:  # and ins.target.is_stackvar:
                self.constants[ins.lhs] = self.constants[ins.rhs]
            elif ins.lhs in self.constants:
                del self.constants[ins.lhs]
        else:
            for var in tac.gens(ins):
                if var in self.constants:
                    del self.constants[ins.lhs]

    def __str__(self) -> str:
        return 'ConstantDomain({})'.format(self.constants)


def hardcode_constants(ins, constants) -> Tac:
    return ins._replace(uses=tuple(uses))
