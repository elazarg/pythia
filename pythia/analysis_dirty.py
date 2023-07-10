import typing
from typing import TypeAlias

from pythia import tac, analysis_domain as domain
from pythia.graph_utils import Location
from pythia.analysis_allocation import AllocationType
from pythia.analysis_domain import InvariantMap, InstructionLattice, Bottom, BOTTOM, VarMapDomain
from pythia.analysis_liveness import Liveness
from pythia.analysis_typed_pointer import TypedPointer, Object, LOCALS, find_reachable

Dirty: TypeAlias = set[Object] | Bottom


class DirtyLattice(InstructionLattice[Dirty]):
    backward: bool = False
    typed_pointer_map: InvariantMap[TypedPointer]

    def name(self) -> str:
        return "Dirty"

    def __init__(self, typed_pointer_map: InvariantMap[TypedPointer]) -> None:
        self.typed_pointer_map = typed_pointer_map

    def copy(self, values: Dirty) -> Dirty:
        return values.copy()

    def initial(self) -> Dirty:
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
            # case tac.For():
            #     values.clear()
            case tac.Assign() as assign:
                match assign.lhs:
                    case tac.Attribute(var=tac.Var() as var) | tac.Subscript(var=tac.Var() as var):
                        values.update(self.typed_pointer_map[location].pointers[LOCALS][var])
                    case tac.Var() as var:
                        if self.allocation_invariant_map[location] != AllocationType.NONE:
                            values.update(self.typed_pointer_map[location].pointers[LOCALS][var])
        return values


def find_reaching_locals(ptr: TypedPointer, liveness: VarMapDomain[Liveness], dirty_objects: Dirty) -> typing.Iterator[str]:
    assert not isinstance(liveness, domain.Bottom)
    assert not isinstance(dirty_objects, domain.Bottom)

    alive = {k for k, v in liveness.items() if isinstance(v, domain.Top)}
    dirty = {object_to_location(obj) for obj in dirty_objects}
    for k, v in ptr.pointers[LOCALS].items():
        if k.name == 'return':
            continue
        reachable = set(find_reachable(ptr, alive, set(), v))
        if k in alive and reachable & dirty:
            yield k.name
