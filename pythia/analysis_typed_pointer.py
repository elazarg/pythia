from __future__ import annotations as _

import enum
import typing
from dataclasses import dataclass
from typing import TypeAlias, Final

from pythia.analysis_types import TypeLattice
from pythia import tac
from pythia.graph_utils import Location
from pythia import analysis_domain as domain
from pythia import analysis_liveness
from pythia.analysis_domain import InstructionLattice, InvariantMap, BOTTOM, VarMapDomain
from pythia.analysis_liveness import Liveness
import pythia.type_system as ts


# Abstract location can be either:
# 1. line of code where it is allocated
# 2. parameter
# 3. immutable value of a certain type
# 4. Scope: Globals, locals*

@dataclass(frozen=True)
class Param:
    param: typing.Optional[tac.Var] = None

    def __repr__(self) -> str:
        return f'@param {self.param}'


@dataclass(frozen=True)
class Immutable:
    type: ts.TypeExpr

    def __repr__(self) -> str:
        return f'@type {self.type}'


@dataclass(frozen=True)
class Scope:
    name: str

    def __repr__(self) -> str:
        return f'@scope {self.name}'


@dataclass(frozen=True)
class LocationObject:
    location: Location

    def __repr__(self) -> str:
        return f'@location {self.location}'


Object: TypeAlias = typing.Union[LocationObject, Param, Immutable, Scope]


LOCALS: Final[Object] = Scope('locals')
GLOBALS: Final[Object] = Scope('globals')


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

    def __repr__(self):
        return f'Pointer({self.graph})'

    def __init__(self, graph: Graph):
        self.graph = graph.copy()

    def __iter__(self) -> typing.Iterator[Object]:
        return iter(self.graph.keys())

    def items(self) -> typing.Iterable[tuple[Object, Fields]]:
        return self.graph.items()

    def is_less_than(self, other: Pointer) -> bool:
        return all(self.graph[obj][field] <= other.graph[obj][field]
                   for obj in self.graph
                   for field in self.graph[obj])

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
            case (Param() | Immutable() | Scope() | LocationObject() as obj, tac.Var() as var), (frozenset() as values):
                if obj not in self.graph:
                    self.graph[obj] = make_fields({var: values})
                else:
                    self.graph[obj][var] = values
            case Param() | Immutable() | Scope() | LocationObject() as obj, (domain.Map() as values):
                self.graph[obj] = values
            case _:
                raise ValueError(f'Invalid key {key} or value {value}')

    def keep_keys(self, keys: typing.Iterable[Object]) -> None:
        for obj in set(self):
            if obj not in keys:
                del self.graph[obj]

    def __str__(self) -> str:
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

    def __repr__(self):
        return f'TypeMap({self.map})'

    def __init__(self, map: domain.Map[Object, ts.TypeExpr]):
        self.map = map.copy()

    def is_less_than(self, other: TypeMap) -> bool:
        # TODO: check
        return all(ts.is_subtype(self.map[obj], other.map[obj])
                   for obj in self.map)

    def copy(self) -> TypeMap:
        return TypeMap(self.map.copy())

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
            case frozenset():
                return ts.join_all(self[x] for x in obj)
            case obj:
                return self.map[obj]

    def __setitem__(self, key: Object | frozenset[Object], value: ts.TypeExpr) -> None:
        match key:
            case frozenset():
                for x in key:
                    self.map[x] = value
            case obj:
                self.map[obj] = value

    def keep_keys(self, keys: typing.Iterable[Object]) -> None:
        for obj in set(self):
            if obj not in keys:
                del self.map[obj]

    def __iter__(self) -> typing.Iterator[Object]:
        return iter(self.map)

    def join(self, other: TypeMap) -> TypeMap:
        left = self.map
        right = other.map
        res: domain.Map[Object, ts.TypeExpr] = TypeMap.bottom().map
        for k in left.keys() | right.keys():
            res[k] = ts.join(left[k], right[k])
        return TypeMap(res)

    def __str__(self) -> str:
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

    def __repr__(self):
        return f'TP:\n {self.pointers}\n {self.types}\n'

    def is_less_than(self: TypedPointer, other: TypedPointer) -> bool:
        return self.pointers.is_less_than(other.pointers) and self.types.is_less_than(other.types)

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

    def __str__(self) -> str:
        return str(self.pointers) + '\n' + str(self.types)

    @staticmethod
    def initial(annotations: domain.Map[Object, ts.TypeExpr]) -> TypedPointer:
        return typed_pointer(Pointer.initial(annotations),
                             TypeMap.initial(annotations))

    def collect_garbage(self, alive) -> None:
        if isinstance(alive, domain.Bottom):
            return

        for var in set(self.pointers[LOCALS].keys()):
            if var.is_stackvar:
                if alive[var] == BOTTOM:
                    del self.pointers[LOCALS][var]

        reachable = find_reachable_from_vars(self.pointers)
        self.pointers.keep_keys(reachable)
        self.types.keep_keys(reachable)


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
    annotations = {Param(tac.Var(row.index.name)): row.type
                   for row in this_signature.params.row_items()
                   if row.index.name is not None}
    # annotations[Param(tac.Var('return'))] = this_signature.return_type
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

    def __init__(self,
                 liveness: InvariantMap[VarMapDomain[analysis_liveness.Liveness]],
                 this_function: str, this_module: ts.Module) -> None:
        super().__init__()
        self.annotations = parse_annotations(this_function, this_module)
        self.type_lattice = TypeLattice(this_function, this_module)
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

    def expr(self, prev_tp: TypedPointer, expr: tac.Expr, location: LocationObject,
             new_tp: TypedPointer) -> tuple[frozenset[Object], ts.TypeExpr]:
        match expr:
            case tac.Const(value):
                t = self.type_lattice.const(value)
                return (frozenset({Immutable(t)}), t)
            case tac.Var() as var:
                objs = prev_tp.pointers[LOCALS][var]
                types = prev_tp.types[objs]
                return (objs, types)
            case tac.Attribute(var=tac.Predefined.GLOBALS, field=tac.Var() as field):
                global_objs = prev_tp.pointers[GLOBALS][field]
                assert not global_objs
                t = ts.resolve_static_ref(ts.Ref(f'builtins.{field}'))
                # TODO: class through type
                return (frozenset({Immutable(t)}), t)
            case tac.Attribute(var=tac.Var() as var, field=tac.Var() as field):
                var_objs = prev_tp.pointers[LOCALS][var]
                t = self.type_lattice.attribute(prev_tp.types[var_objs], field)
                direct_objs = flatten(prev_tp.pointers[var_obj][field] for var_obj in var_objs)
                if ts.is_bound_method(t):
                    new_tp.pointers[location, tac.Var("self")] = var_objs
                    return (frozenset({location}) | direct_objs, t)
                # TODO: class through type
                return (direct_objs, t)
            case tac.Subscript(var=tac.Var() as var, index=tac.Var() as index):
                var_objs = prev_tp.pointers[LOCALS][var]
                index_objs = prev_tp.pointers[LOCALS][index]
                direct_objs = flatten(prev_tp.pointers[var_obj][tac.Var("*")] for var_obj in var_objs)
                types = self.type_lattice.subscr(prev_tp.types[var_objs], prev_tp.types[index_objs])
                # TODO: class through type
                return (direct_objs, types)
            case tac.Call(var, tuple() as args):
                if isinstance(var, tac.Var):
                    func_objects = prev_tp.pointers[LOCALS][var]
                    func_type = prev_tp.types[func_objects]
                else:
                    func_objects = frozenset({})
                    func_type = self.type_lattice.predefined(var)
                assert isinstance(func_type, ts.Overloaded), f"Expected Overloaded type, got {func_type}"
                arg_objects = [prev_tp.pointers[LOCALS][var] for var in args]
                arg_types = tuple([prev_tp.types[obj] for obj in arg_objects])
                applied = ts.partial_positional(func_type, arg_types)
                assert isinstance(applied, ts.Overloaded), f"Expected Overloaded type, got {applied}"
                side_effect = ts.get_side_effect(applied)
                if side_effect.update is not None:
                    if len(func_objects) != 1:
                        raise RuntimeError("Cannot handle update with multiple function objects")
                    [func_obj] = func_objects
                    self_objects = prev_tp.pointers[func_obj][tac.Var("self")]
                    if len(self_objects) != 1:
                        raise RuntimeError("Cannot handle update with multiple self objects")
                    [self_obj] = self_objects

                    aliasing_pointers = {obj for obj, fields in prev_tp.pointers.items() for f, targets in fields.items()
                                         if self_obj in targets} - {func_obj, LOCALS}
                    monomorophized = [obj for obj in aliasing_pointers if ts.is_monomorphized(prev_tp.types[obj])]
                    # Expected two objects: self argument and locals
                    if monomorophized:
                        raise RuntimeError(f"Cannot handle update with aliased objects: {aliasing_pointers}")
                    new_tp.types[self_obj] = side_effect.update

                t = ts.get_return(applied)
                objects: frozenset[Object] = frozenset([])
                if any(f.new() for f in applied.items):
                    objects |= frozenset([location])
                    if side_effect.points_to_args:
                        new_tp.pointers[location, tac.Var("*")] = frozenset(ts.union_all(set(x) for x in arg_objects))
                elif ts.is_immutable(t):
                    objects |= frozenset({Immutable(t)})

                return (objects, t)
            case tac.Unary(var=tac.Var() as var, op=tac.UnOp() as op):
                value_objects = prev_tp.pointers[LOCALS][var]
                assert value_objects, f"Expected objects for {var}"
                arg_type = prev_tp.types[value_objects]
                applied = ts.get_unop(arg_type, self.type_lattice.unop_to_str(op))
                assert isinstance(applied, ts.Overloaded), f"Expected overloaded type, got {applied}"

                t = ts.get_return(applied)
                objects = frozenset({})
                if any(f.new() for f in applied.items):
                    objects |= frozenset([location])
                elif ts.is_immutable(t):
                    objects |= frozenset({Immutable(t)})

                return (objects, t)
            case tac.Binary(left=tac.Var() as left, right=tac.Var() as right, op=str() as op):
                left_objects = prev_tp.pointers[LOCALS][left]
                right_objects = prev_tp.pointers[LOCALS][right]
                left_type = prev_tp.types[left_objects]
                right_type = prev_tp.types[right_objects]
                applied = ts.partial_binop(left_type, right_type, op)
                assert isinstance(applied, ts.Overloaded), f"Expected overloaded type, got {applied}"

                t = ts.get_return(applied)
                objects = frozenset({})
                if any(f.new() for f in applied.items):
                    objects |= frozenset([location])
                elif ts.is_immutable(t):
                    objects |= frozenset({Immutable(t)})

                return (objects, t)
            case _:
                raise NotImplementedError(expr)
        assert False

    def signature(self, tp: TypedPointer, signature: tac.Signature, pointed: frozenset[Object], t: ts.TypeExpr) -> None:
        match signature:
            case tuple() as signature:  # type: ignore
                raise NotImplementedError
            case tac.Var() as var:
                tp.pointers[LOCALS, var] = pointed
                tp.types[pointed] = t
            case tac.Attribute(var, field):
                for obj in tp.pointers[LOCALS][var]:
                    tp.pointers[obj, field] = pointed
            case tac.Subscript(var, index):
                for obj in tp.pointers[LOCALS][var]:
                    tp.pointers[obj, tac.Var('*')] = pointed
            case _:
                assert False, f'unexpected signature {signature}'

    def transfer(self, prev_tp: TypedPointer, ins: tac.Tac, _location: Location) -> TypedPointer:
        tp = prev_tp.copy()
        location = LocationObject(_location)

        if isinstance(ins, tac.For):
            ins = ins.as_call()

        # FIX: this removes pointers and make it "bottom" instead of "top"
        for var in tac.gens(ins):
            if var in tp.pointers[LOCALS]:
                del tp.pointers[LOCALS][var]

        match ins:
            case tac.Assign(lhs, expr):
                (pointed, types) = self.expr(prev_tp, expr, location, tp)
                self.signature(tp, lhs, pointed, types)
            case tac.Return(var):
                val = tp.pointers[LOCALS][var]
                tp.pointers[LOCALS, tac.Var('return')] = val

        tp.collect_garbage(self.liveness[location.location])

        return tp


def find_reachable_from_vars(ptr: Pointer) -> set[Object]:
    worklist = {LOCALS}
    reachable = set(worklist)
    while worklist:
        root = worklist.pop()
        for edge, objects in ptr[root].items():
            worklist.update(objects)
            reachable.update(objects)
    return reachable


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
