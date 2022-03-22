# Data flow analysis and stuff.

from __future__ import annotations

import typing

import tac
from tac import Tac
from tac_analysis_domain import AbstractDomain

T = typing.TypeVar('T')


class ConstantDomain(AbstractDomain):
    def __init__(self, constants: typing.Optional[dict[tac.Var, tac.Const]] = ()) -> None:
        super().__init__()
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
    def top(cls: T) -> T:
        return ConstantDomain({})

    def set_to_top(self) -> None:
        self.constants = {}

    @classmethod
    def bottom(cls: T) -> T:
        return ConstantDomain(None)

    def join(self: T, other: T) -> T:
        if self.constants is None:
            return other.copy()
        if other.constants is None:
            return self.copy()
        return ConstantDomain(dict(self.constants.items() & other.constants.items()))

    def single_block_update(self, block: list[tac.Tac]) -> None:
        if self.is_bottom:
            return
        for i, ins in enumerate(block):
            if isinstance(ins, tac.Assign) and isinstance(ins.target, tac.Var):
                if isinstance(ins.value, tac.Const):
                    self.constants[ins.target] = ins.value
                elif ins.value in self.constants:  # and ins.target.is_stackvar:
                    self.constants[ins.target] = self.constants[ins.value]

    @property
    def is_bottom(self):
        return self.constants is None


def hardcode_constants(ins, constants) -> Tac:
    return ins._replace(uses=tuple(uses))
