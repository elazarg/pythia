from __future__ import annotations as _

import typing
from dataclasses import dataclass

from pythia import tac
from pythia.graph_utils import Label, Location


class Lattice[T](typing.Protocol):

    @classmethod
    def name(cls) -> str:
        raise NotImplementedError

    @classmethod
    def top(cls) -> T:
        raise NotImplementedError

    @classmethod
    def bottom(cls) -> T:
        raise NotImplementedError

    @classmethod
    def is_bottom(cls, elem: T) -> bool:
        raise NotImplementedError

    def join(self, left: T, right: T) -> T:
        raise NotImplementedError

    def join_all(self, inv: T, *invs: T) -> T:
        for arg in invs:
            inv = self.join(inv, arg)
        return inv

    def is_less_than(self, left: T, right: T) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class Top:
    def __str__(self) -> str:
        return "⊤"

    def __deepcopy__(self, memodict={}):
        return self

    def __or__(self, other: object) -> Top:
        return self

    def __ror__(self, other: object) -> Top:
        return self

    def __and__[T](self, other: T) -> T:
        return other

    def __rand__[T](self, other: T) -> T:
        return other

    def __contains__(self, item: object) -> bool:
        return True

    def __copy__(self):
        return self


@dataclass(frozen=True)
class Bottom:
    def __str__(self) -> str:
        return "⊥"

    def __deepcopy__(self, memodict={}):
        memodict[id(self)] = self
        return self

    def __or__[T](self, other: T) -> T:
        return other

    def __ror__[T](self, other: T) -> T:
        return other

    def __rand__(self, other: object) -> Bottom:
        return self

    def __and__(self, other: object) -> Bottom:
        return self


BOTTOM = Bottom()
TOP = Top()


class InstructionLattice[T](Lattice[T], typing.Protocol):
    backward: bool

    def transfer(self, values: T, ins: tac.Tac, location: Location) -> T:
        raise NotImplementedError

    def initial(self) -> T:
        return self.top()


class ValueLattice[T](Lattice[T], typing.Protocol):
    def const(self, value: int | str | bool | float | tuple | list | None) -> T:
        return self.top()

    def var(self, value: T) -> T:
        return value

    def attribute(self, var: T, attr: tac.Var) -> T:
        assert isinstance(attr, tac.Var)
        return self.top()

    def subscr(self, array: T, index: T) -> T:
        return self.top()

    def call(self, function: T, args: list[T]) -> T:
        return self.top()

    def predefined(self, name: tac.PredefinedFunction) -> T:
        return self.top()

    def annotation(self, name: tac.Var, t: str) -> T:
        return self.top()

    def default(self) -> T:
        return self.top()


type InvariantMap[T] = dict[Label, T]
