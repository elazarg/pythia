from __future__ import annotations

from typing import TypeAlias
import tac
from graph_utils import Location
from analysis_domain import InvariantMap, InstructionLattice, Bottom, BOTTOM
from analysis_pointer import Graph, Object, LOCALS

Dirty: TypeAlias = set[Object] | Bottom


class DirtyLattice(InstructionLattice[Dirty]):
    backward: bool = False
    pointer_map: InvariantMap[Graph]

    def name(self) -> str:
        return "Dirty"

    def __init__(self, pointer_map: InvariantMap[Graph]) -> None:
        self.pointer_map = pointer_map

    def copy(self, values: Dirty) -> Dirty:
        return values.copy()

    def initial(self, annotations: dict[tac.Var, str]) -> Dirty:
        return set()

    def bottom(self) -> Dirty:
        return BOTTOM

    def top(self) -> Dirty:
        raise NotImplementedError

    def is_top(self, elem: Dirty) -> bool:
        return False

    def is_bottom(self, elem: Dirty) -> bool:
        return elem == self.bottom()

    def is_less_than(self, left: Dirty, right: Dirty) -> bool:
        return self.join(left, right) == right

    def is_equivalent(self, left: Dirty, right: Dirty) -> bool:
        return left == right

    def join(self, left: Dirty, right: Dirty) -> Dirty:
        if left is BOTTOM:
            return right
        if right is BOTTOM:
            return left
        return left | right

    def transfer(self, values: Dirty, ins: tac.Tac, location: Location) -> Dirty:
        if isinstance(values, Bottom):
            return self.bottom()
        values = values.copy()
        match ins:
            case tac.Assign(lhs=tac.Attribute(var=tac.Var() as var) | tac.Subscript(var=tac.Var() as var)):
                values.update(self.pointer_map[location][LOCALS][var])
        return values
