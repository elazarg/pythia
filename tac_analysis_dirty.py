from __future__ import annotations

from typing import TypeAlias
import tac
from tac_analysis_domain import Lattice
from tac_analysis_pointer import Graph, Object, LOCALS

Dirty: TypeAlias = set[Object]


def copy_dirty(dirty: Dirty) -> Dirty:
    return dirty.copy()


class DirtyLattice(Lattice[Dirty]):
    backward: bool = False

    def name(self) -> str:
        return "Dirty"

    def __init__(self) -> None:
        super().__init__()

    def is_less_than(self, left: Dirty, right: Dirty) -> bool:
        return self.join(left, right) == right

    def is_equivalent(self, left: Dirty, right: Dirty) -> bool:
        return left == right

    def copy(self, values: Dirty) -> Dirty:
        return copy_dirty(values)

    def initial(self, annotations: dict[tac.Var, str]) -> Dirty:
        return set()

    def bottom(self) -> Dirty:
        return set()

    def join(self, left: Dirty, right: Dirty) -> Dirty:
        return left | right

    def transfer(self, values: Dirty, ins: tac.Tac, pointers: Graph, location: str) -> Dirty:
        values = values.copy()
        match ins:
            case tac.Assign(lhs=tac.Attribute(var=tac.Var() as var) | tac.Subscript(var=tac.Var() as var)):
                values.update(pointers[LOCALS][var])
        return values
