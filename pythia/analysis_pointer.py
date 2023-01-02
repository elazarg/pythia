# Data flow analysis and stuff.

from __future__ import annotations as _

import typing
from dataclasses import dataclass
from itertools import chain
from typing import TypeAlias, Final

from pythia import tac
from pythia.graph_utils import Location
from . import analysis_domain as domain
from . import analysis_liveness
from .analysis_domain import InstructionLattice, InvariantMap, BOTTOM, MapDomain
from .analysis_types import AllocationType


@dataclass(frozen=True)
class Object:
    location: str

    def __str__(self) -> str:
        return f'@{self.location}'

    def __repr__(self) -> str:
        return f'@{self.location}'


LOCALS: Final[Object] = Object('locals()')
NONLOCALS: Final[Object] = Object('NONLOCALS')

GLOBALS: Final[Object] = Object('globals()')


Graph: TypeAlias = dict[Object, dict[tac.Var, frozenset[Object]]]


def pretty_print_pointers(pointers: Graph) -> str:
    join = lambda target_obj: "{" + ", ".join(str(x) for x in target_obj) + "}"
    return ', '.join((f'{source_obj}:' if source_obj is not LOCALS else '') + f'{field}->{join(target_obj)}'
                     for source_obj in pointers
                     for field, target_obj in pointers[source_obj].items()
                     if pointers[source_obj][field]
                     )

def invert_graph(graph: Graph) -> Graph:
    result = {}
    for obj, obj_fields in graph.items():
        for field, target_obj in obj_fields.items():
            for target in target_obj:
                result.setdefault(target, {}).setdefault(field, set()).add(obj)
    return result


def copy_graph(graph: Graph) -> Graph:
    return {obj: {field: target_obj.copy() for field, target_obj in obj_fields.items()}
            for obj, obj_fields in graph.items()}


class PointerLattice(InstructionLattice[Graph]):
    allocation_invariant_map: InvariantMap[AllocationType]
    liveness: InvariantMap[MapDomain[analysis_liveness.Liveness]]
    backward: bool = False

    def name(self) -> str:
        return "Pointer"

    def __init__(self, allocation_invariant_map: InvariantMap[AllocationType], liveness: InvariantMap[MapDomain[
        analysis_liveness.Liveness]]) -> None:
        super().__init__()
        self.allocation_invariant_map = allocation_invariant_map
        self.liveness = liveness
        self.backward = False

    def is_less_than(self, left: Graph, right: Graph) -> bool:
        return self.join(left, right) == right

    def is_equivalent(self, left: Graph, right: Graph) -> bool:
        return left == right

    def copy(self, values: Graph) -> Graph:
        return {obj: {field: targets.copy() for field, targets in fields.items() if targets}
                for obj, fields in values.items()}

    def initial(self, annotations: dict[tac.Var, str]) -> Graph:
        return {LOCALS: {k: frozenset({Object(f'param {k}')}) for k in annotations},
                GLOBALS: {}}

    def bottom(self) -> Graph:
        return {}

    def top(self) -> Graph:
        raise NotImplementedError

    def is_top(self, elem: Graph) -> bool:
        return False

    def is_bottom(self, elem: Graph) -> bool:
        return elem == self.bottom()

    def join(self, left: Graph, right: Graph) -> Graph:
        pointers = copy_graph(left)
        for obj, fields in right.items():
            if obj in pointers:
                for field, values in fields.items():
                    pointers[obj][field] = pointers[obj].get(field, frozenset()) | values
            else:
                pointers[obj] = {field: targets.copy() for field, targets in fields.items() if targets}
        return pointers

    def transfer(self, values: Graph, ins: tac.Tac, location: Location) -> Graph:
        values = copy_graph(values)
        location_object = Object(f'{location[0]}.{location[1]}')
        allocated = self.allocation_invariant_map.get(location) != AllocationType.NONE

        def eval(expr: tac.Expr) -> frozenset[Object]:
            match expr:
                case tac.Var():
                    return values[LOCALS].get(expr, frozenset()).copy()
                case tac.Call() | tac.Unary() | tac.Binary():
                    if allocated:
                        return frozenset({location_object})
                    return frozenset()
                case tac.Attribute():
                    if allocated:
                        return frozenset({location_object})
                    if expr.var.name == 'GLOBALS':
                        return values[GLOBALS].get(expr.field, frozenset()).copy()
                    else:
                        return frozenset(chain.from_iterable(values.get(obj, {}).get(expr.field, frozenset()).copy()
                                                             for obj in eval(expr.var)))
                case _: return frozenset()

        activation = values[LOCALS]

        for var in tac.gens(ins):
            if var in activation:
                del activation[var]

        match ins:
            case tac.Assign():
                val = eval(ins.expr)
                match ins.lhs:
                    case tac.Var():
                        activation[ins.lhs] = val
                    case tac.Attribute():
                        for obj in eval(ins.lhs.var):
                            values.setdefault(obj, {})[ins.lhs.field] = val
                    case tac.Subscript():
                        for obj in eval(ins.lhs.var):
                            values.setdefault(obj, {})[tac.Var('*')] = val
            case tac.Return():
                val = eval(ins.value)
                activation[tac.Var('return')] = val

        here = self.liveness[location]
        if isinstance(here, domain.Bottom):
            return values

        for var in set(activation.keys()):
            if var.is_stackvar and here[var] is BOTTOM:
                del activation[var]

        return values

def allocation_to_str(t: AllocationType) -> str:
    if t is not AllocationType.NONE:
        return f' #  ' + t.name
    return ''


def find_reachable(ptr: Graph, alive: set[tac.Var], annotations: dict[tac.Var, object],
                   sources: typing.Optional[frozenset[Object]] = None) -> typing.Iterator[Location]:
    worklist = set(sources) if sources is not None else {LOCALS}
    while worklist:
        root = worklist.pop()
        for edge, locs in ptr.get(root, {}).items():
            if root == LOCALS and edge not in alive:
                # We did not remove non-stack variables from the pointer lattice, so we need to filter them out here.
                continue
            if edge in annotations:
                continue
            for loc in locs:
                if repr(loc).startswith('@param'):
                    continue
                worklist.add(loc)
                label, index = [int(x) for x in str(loc)[1:].split('.')]
                yield (label, index)

def find_reaching_locals(ptr: Graph, alive: set[tac.Var], dirty: set[Object]) -> typing.Iterator[tac.Var]:
    for k, v in ptr[LOCALS].items():
        if k in alive and v & dirty:
            yield k
