# Data flow analysis and stuff.

from __future__ import annotations

from dataclasses import dataclass
from typing import Type, TypeVar, Optional, ClassVar

import tac
from tac import Const, Var
from tac_analysis_domain import AbstractDomain, IterationStrategy, ForwardIterationStrategy

import graph_utils as gu

T = TypeVar('T')


@dataclass
class Bottom:
    pass


class ConstantDomain(AbstractDomain):
    constants: dict[Var, Const] | Bottom

    BOTTOM: ClassVar[Bottom]

    @staticmethod
    def name() -> str:
        return "Constant"

    @classmethod
    def view(cls, cfg: gu.Cfg[T]) -> IterationStrategy[T]:
        return ForwardIterationStrategy(cfg)

    def __init__(self, constants: Optional[dict[Var, Const]] = ()) -> None:
        super().__init__()
        if constants is None:
            self.constants = ConstantDomain.BOTTOM
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

    def __repr__(self) -> str:
        return self.constants.__repr__()


ConstantDomain.BOTTOM = Bottom()


def hardcode_constants(ins, constants) -> tac.Tac:
    return ins._replace(uses=tuple(uses))
