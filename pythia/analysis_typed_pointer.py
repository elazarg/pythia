from __future__ import annotations as _

import enum
import typing
from dataclasses import dataclass
from itertools import chain
from typing import TypeAlias, Final

from analysis_types import TypeLattice
from pythia import tac
from pythia.graph_utils import Location
from pythia import analysis_domain as domain
from pythia import analysis_liveness
from pythia.analysis_domain import InstructionLattice, InvariantMap, BOTTOM, VarMapDomain
from pythia.analysis_liveness import Liveness
import pythia.type_system as ts


@dataclass(frozen=True)
class Object:
    location: str
    param: typing.Optional[tac.Var] = None

    def __repr__(self) -> str:
        return f'@{self.location}{self.param if self.param else ""}'


def param_object(p: tac.Var) -> Object:
    return Object('param ', p)


LOCALS: Final[Object] = Object('locals()')
NONLOCALS: Final[Object] = Object('NONLOCALS')

GLOBALS: Final[Object] = Object('globals()')

Fields: TypeAlias = domain.Map[tac.Var, frozenset[Object]]
Graph: TypeAlias = domain.Map[Object, Fields]


def make_fields(d: typing.Optional[dict[tac.Var, frozenset[Object]]] = None) -> Fields:
    d = d or {}
    return domain.Map(default=frozenset(), d=d)


def make_graph(d: typing.Optional[dict[Object, Fields]] = None) -> Graph:
    d = d or {}
    return domain.Map(default=make_fields(), d=d)


def make_type_map(d: typing.Optional[dict[Object, ts.TypeExpr]] = None) -> domain.Map[Object, ts.TypeExpr]:
    d = d or {}
    return domain.Map(default=ts.BOTTOM, d=d)


class Pointer:
    graph: Graph

    def __init__(self, graph: Graph):
        self.graph = graph.copy()

    def is_less_than(self, other: Pointer) -> bool:
        return self.join(other) == other

    def copy(self) -> Pointer:
        return Pointer(self.graph)

    @staticmethod
    def bottom() -> Pointer:
        return Pointer(make_graph())

    @staticmethod
    def top() -> Graph:
        raise NotImplementedError

    def is_top(self) -> bool:
        return False

    def is_bottom(self) -> bool:
        return False

    def join(self, other: Pointer) -> Pointer:
        pointers = self.graph.copy()
        for obj, fields in other.graph.items():
            if obj in pointers:
                for field, values in fields.items():
                    pointers[obj][field] = pointers[obj][field] | values
            else:
                pointers[obj] = make_fields({field: targets.copy()
                                             for field, targets in fields.items() if targets})
        return Pointer(pointers)

    def __getitem__(self, obj: Object) -> Fields:
        return self.graph[obj]

    def __setitem__(self, key: Object | tuple[Object, tac.Var], value: Fields | frozenset[Object]) -> None:
        match key, value:
            case (Object() as obj, tac.Var() as var), (frozenset() as values):
                self.graph[obj][var] = values
            case Object() as obj, (domain.Map() as values):
                self.graph[obj] = values
            case _:
                raise ValueError(f'Invalid key {key} or value {value}')

    def pretty_print(self) -> str:
        join = lambda target_obj: "{" + ", ".join(str(x) for x in target_obj) + "}"
        return ', '.join((f'{source_obj}:' if source_obj is not LOCALS else '') + f'{field}->{join(target_obj)}'
                         for source_obj in self.graph
                         for field, target_obj in self.graph[source_obj].items()
                         if self.graph[source_obj][field]
                         )

    @staticmethod
    def initial(annotations: domain.Map[Object, ts.TypeExpr]) -> Pointer:
        return Pointer(make_graph({LOCALS: make_fields({obj.param: frozenset({obj}) for obj in annotations}),
                                   GLOBALS: make_fields()}))


class TypeMap:
    map: domain.Map[Object, ts.TypeExpr]

    def __init__(self, map: domain.Map[Object, ts.TypeExpr]):
        self.map = map.copy()

    def is_less_than(self, other: TypeMap) -> bool:
        return self.join(other) == other

    def copy(self) -> TypeMap:
        return self

    @staticmethod
    def bottom() -> TypeMap:
        return TypeMap(make_type_map())

    @staticmethod
    def top() -> TypeMap:
        raise NotImplementedError

    def is_top(self) -> bool:
        return False

    def is_bottom(self) -> bool:
        return False

    def __getitem__(self, obj: Object | frozenset[Object]) -> ts.TypeExpr:
        match obj:
            case Object(): return self.map[obj]
            case frozenset(): return ts.join_all(self[x] for x in obj)

    def join(self, other: TypeMap) -> TypeMap:
        left = self.map
        right = other.map
        res: domain.Map[Object, ts.TypeExpr] = self.top().map
        for k in left.keys() | right.keys():
            res[k] = ts.join(left[k], right[k])
        return TypeMap(res)

    def pretty_print(self) -> str:
        return ','.join(f'{obj}: {type_expr}' for obj, type_expr in self.map.items())

    @staticmethod
    def initial(annotations: domain.Map[Object, ts.TypeExpr]) -> TypeMap:
        return TypeMap(annotations)


class AllocationType(enum.StrEnum):
    NONE = ''
    STACK = 'Stack'
    HEAP = 'Heap'
    UNKNOWN = 'Unknown'


Allocation: typing.TypeAlias = AllocationType


@dataclass(frozen=True)
class TypedPointer:
    pointers: Pointer
    types: TypeMap

    def is_less_than(self: TypedPointer, other: TypedPointer) -> bool:
        return self.join(other) == other

    def copy(self: TypedPointer) -> TypedPointer:
        return TypedPointer(self.pointers.copy(),
                            self.types.copy())

    @staticmethod
    def bottom() -> TypedPointer:
        return TypedPointer(Pointer.bottom(),
                            TypeMap.bottom())

    @staticmethod
    def top() -> TypedPointer:
        raise NotImplementedError

    def is_top(self) -> bool:
        return self.pointers.is_top() and self.types.is_top()

    def is_bottom(self) -> bool:
        return self.pointers.is_bottom() or self.types.is_bottom()

    def join(self, right: TypedPointer) -> TypedPointer:
        return typed_pointer(self.pointers.join(right.pointers),
                             self.types.join(right.types))

    def pretty_print(self) -> str:
        return self.pointers.pretty_print() + '\n' + self.types.pretty_print()

    @staticmethod
    def initial(annotations: domain.Map[Object, ts.TypeExpr]) -> TypedPointer:
        return typed_pointer(Pointer.initial(annotations),
                             TypeMap.initial(annotations))


def typed_pointer(pointers: Pointer, types: TypeMap) -> TypedPointer:
    # Normalization.
    if pointers.is_bottom() or types.is_bottom():
        return TypedPointer.bottom()
    return TypedPointer(pointers, types)


def parse_annotations(this_function: str, this_module: ts.Module) -> TypeMap:
    this_signature = ts.subscr(this_module, ts.literal(this_function))
    assert isinstance(this_signature, ts.Overloaded), f"Expected overloaded type, got {this_signature}"
    assert len(this_signature.items) == 1, f"Expected single signature, got {this_signature}"
    [this_signature] = this_signature.items
    annotations = {param_object(tac.Var(row.index.name)): row.type
                   for row in this_signature.params.row_items()
                   if row.index.name is not None}
    annotations[Object('return')] = this_signature.return_type
    return TypeMap(make_type_map(annotations))


def flatten(xs: typing.Iterable[frozenset[Object]]) -> frozenset[Object]:
    return frozenset(obj for x in xs for obj in x)


class TypedPointerLattice(InstructionLattice[TypedPointer]):
    type_lattice: TypeLattice
    liveness: InvariantMap[VarMapDomain[analysis_liveness.Liveness]]
    annotations: TypeMap
    backward: bool = False

    def name(self) -> str:
        return "TypedPointer"

    def __init__(self, type_lattice: TypeLattice,
                 liveness: InvariantMap[VarMapDomain[analysis_liveness.Liveness]],
                 this_function: str, this_module: ts.Module) -> None:
        super().__init__()
        self.annotations = parse_annotations(this_function, this_module)
        self.type_lattice = type_lattice
        self.liveness = liveness
        self.backward = False

    def is_less_than(self, left: TypedPointer, right: TypedPointer) -> bool:
        return left.is_less_than(right)

    def copy(self, tp: TypedPointer) -> TypedPointer:
        return tp.copy()
    
    def initial(self) -> TypedPointer:
        return TypedPointer.initial(self.annotations.map)

    def bottom(self) -> TypedPointer:
        return TypedPointer.bottom()

    def top(self) -> TypedPointer:
        raise NotImplementedError

    def is_top(self, tp: TypedPointer) -> bool:
        return False

    def is_bottom(self, tp: TypedPointer) -> bool:
        return tp.is_bottom()

    def join(self, left: TypedPointer, right: TypedPointer) -> TypedPointer:
        return left.join(right)

    def expr(self, tp: TypedPointer, expr: tac.Expr, location: Object) -> frozenset[Object]:
        match expr:
            case tac.Var() as var:
                objs = tp.pointers[LOCALS][var]
                return objs
            case tac.Attribute(var=tac.Var() as var, field=tac.Var() as field):
                var_objs = tp.pointers[LOCALS][var]
                direct_objs = [tp.pointers[var_obj][field] for var_obj in var_objs]
                # TODO: class through type
                return flatten(direct_objs)
            case tac.Subscript(var=tac.Var() as var, index=tac.Var() as index):
                var_objs = tp.pointers[LOCALS][var]
                direct_objs = [tp.pointers[var_obj][tac.Var("*")] for var_obj in var_objs]
                # TODO: class through type
                return flatten(direct_objs)
            case tac.Call(tac.Var() as var, tuple() as args):
                objects = tp.pointers[LOCALS][var]
                func_type = tp.types[objects]
                arg_objects = [tp.pointers[LOCALS][var] for var in args]
                arg_types = tuple([tp.types[obj] for obj in arg_objects])
                applied = ts.partial_positional(func_type, arg_types)
                assert isinstance(applied, ts.FunctionType)
                if applied.new():
                    return frozenset([location])
                return frozenset()
            case tac.Call(tac.Predefined() as func, tuple() as args):
                raise NotImplementedError
            case tac.Unary(var=tac.Var() as var, op=tac.UnOp() as op):
                raise NotImplementedError
            case tac.Binary(left=tac.Var() as left, right=tac.Var() as right, op=str() as op):
                raise NotImplementedError
        assert False

    def transfer(self, tp: TypedPointer, ins: tac.Tac, location: Location) -> TypedPointer:
        tp = tp.copy()
        location_object = Object(f'{location[0]}.{location[1]}')
        allocation = AllocationType.NONE
        if self.is_bottom(tp):
            allocation = Allocation.UNKNOWN
        if isinstance(ins, tac.Assign):
            self.expr(ins.expr)
        def eval(expr: tac.Expr) -> frozenset[Object]:
            match expr:
                case tac.Var():
                    return tp.pointers[LOCALS][expr]
                case tac.Call() | tac.Unary() | tac.Binary():
                    if allocated:
                        return frozenset({location_object})
                    return frozenset()
                case tac.Attribute():
                    if allocated:
                        return frozenset({location_object})
                    if expr.var.name == 'GLOBALS':
                        return tp.pointers[GLOBALS][expr.field]
                    else:
                        return frozenset(chain.from_iterable(tp.pointers[obj][expr.field]
                                                             for obj in eval(expr.var)))
                case tac.Subscript():
                    if allocated:
                        return frozenset({location_object})
                    return frozenset()
                case _: return frozenset()

        activation = tp.pointers[LOCALS]

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
                            tp.pointers[obj, ins.lhs.field] = val
                    case tac.Subscript():
                        for obj in eval(ins.lhs.var):
                            tp.pointers[obj, tac.Var('*')] = val
            case tac.Return():
                val = eval(ins.value)
                activation[tac.Var('return')] = val

        here = self.liveness[location]
        if isinstance(here, domain.Bottom):
            return tp

        for var in set(activation.keys()):
            if var.is_stackvar:
                if here[var] == BOTTOM:
                    del activation[var]

        return tp


def object_to_location(obj: Object) -> Location:
    label, index = obj.location.split('.')
    return (int(label), int(index))


def find_reachable(ptr: TypedPointer, alive: set[tac.Var], params: set[tac.Var],
                   sources: typing.Optional[frozenset[Object]] = None) -> typing.Iterator[Location]:
    worklist = set(sources) if sources is not None else {LOCALS}
    while worklist:
        root = worklist.pop()
        if '.' in root.location:
            yield object_to_location(root)
        for edge, objects in ptr[root].items():
            if root == LOCALS and edge not in alive:
                # We did not remove non-stack variables from the pointer lattice, so we need to filter them out here.
                continue
            if edge in params:
                continue
            for obj in objects:
                if repr(obj).startswith('@param'):
                    continue
                worklist.add(obj)


def update_allocation_invariants(allocation_invariants: InvariantMap[AllocationType],
                                 ptr: TypedPointer,
                                 liveness: VarMapDomain[Liveness],
                                 annotations: dict[tac.Var, str]) -> None:
    assert not isinstance(liveness, domain.Bottom)

    alive = set(liveness.keys())
    for location in find_reachable(ptr, alive, set(annotations)):
        allocation_invariants[location] = AllocationType.HEAP
