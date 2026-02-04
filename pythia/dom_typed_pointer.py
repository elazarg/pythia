from __future__ import annotations as _

import typing
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Final

import pythia.dom_concrete
from pythia.dom_liveness import LivenessVarLattice, Liveness
from pythia import tac
from pythia.graph_utils import Location
from pythia import domains as domain
from pythia.domains import InstructionLattice
from pythia.dom_concrete import MapDomain
import pythia.type_system as ts


# Abstract location can be either:
# 1. Line of code where it is allocated
# 2. Parameter
# 3. Immutable value of a certain type
# 4. Scope: Globals, locals*


@dataclass(frozen=True)
class Param:
    param: typing.Optional[tac.Var] = None

    def __repr__(self) -> str:
        return f"@param {self.param}"


@dataclass(frozen=True)
class Immutable:
    hash: int

    def __repr__(self) -> str:
        return f"@type {self.hash}"


def immutable(t: ts.TypeExpr, cache={}) -> pythia.dom_concrete.Set[Object]:
    if t not in cache:
        cache[t] = Immutable(len(cache))
    return pythia.dom_concrete.Set[Object].singleton(cache[t])


@dataclass(frozen=True)
class Scope:
    name: str

    def __repr__(self) -> str:
        return f"@scope {self.name}"


LOCALS: Final[Object] = Scope("locals")
GLOBALS: Final[Object] = Scope("globals")


@dataclass(frozen=True)
class LocationObject:
    location: Location

    def __repr__(self) -> str:
        if len(self.location) == 2:
            label, index = self.location
            return f"@location {label + index}"
        else:
            # Field objects have (label, index, field_name)
            label, index, field_name = self.location
            return f"@location {label + index}.{field_name}"


type Object = typing.Union[LocationObject, Param, Immutable, Scope]

type Fields = pythia.dom_concrete.Map[tac.Var, pythia.dom_concrete.Set[Object]]
type Graph = pythia.dom_concrete.Map[Object, Fields]

type Dirty = pythia.dom_concrete.Map[Object, pythia.dom_concrete.Set[tac.Var]]

# Set of objects that represent bound methods (have "self" bound)
type BoundMethods = pythia.dom_concrete.Set[Object]


def make_bound_methods(
        objects: typing.Optional[typing.Iterable[Object]] = None
) -> BoundMethods:
    if objects is None:
        return pythia.dom_concrete.Set[Object]()
    return pythia.dom_concrete.Set[Object](objects)


def make_fields(
        d: typing.Optional[typing.Mapping[tac.Var, pythia.dom_concrete.Set[Object]]] = None
) -> Fields:
    d = d or {}
    return pythia.dom_concrete.Map(default=pythia.dom_concrete.Set, d=d)


def make_graph(d: typing.Optional[typing.Mapping[Object, Fields]] = None) -> Graph:
    d = d or {}
    return pythia.dom_concrete.Map(default=make_fields, d=d)


def make_dirty(
        d: typing.Optional[typing.Mapping[Object, typing.Iterable[tac.Var]]] = None
) -> Dirty:
    d = d or {}
    return pythia.dom_concrete.Map(
        default=pythia.dom_concrete.Set,
        d={k: pythia.dom_concrete.Set(v) for k, v in d.items()},
    )


def make_dirty_from_keys(
        keys: pythia.dom_concrete.Set[Object], field: pythia.dom_concrete.Set[tac.Var]
) -> Dirty:
    return pythia.dom_concrete.Map(
        default=pythia.dom_concrete.Set, d={k: field for k in keys.as_set()}
    )


def make_bottom():
    return ts.BOTTOM


def make_type_map(
        d: typing.Optional[typing.Mapping[Object, ts.TypeExpr]] = None
) -> pythia.dom_concrete.Map[Object, ts.TypeExpr]:
    d = d or {}
    return pythia.dom_concrete.Map(default=make_bottom, d=d)


@dataclass
class Pointer:
    graph: Graph

    def __repr__(self):
        return f"Pointer({self.graph})"

    def __init__(self, graph: Graph):
        self.graph = graph

    def __iter__(self) -> typing.Iterator[Object]:
        return iter(self.graph.keys())

    def items(self) -> typing.Iterable[tuple[Object, Fields]]:
        return self.graph.items()

    def is_less_than(self, other: Pointer) -> bool:
        # Only check keys in self because missing keys return empty set (BOTTOM),
        # and empty.is_subset(x) is always True
        return all(
            self.graph[obj][field].is_subset(other.graph[obj][field])
            for obj in self.graph
            for field in self.graph[obj]
        )

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return Pointer(deepcopy(self.graph, memodict))

    @staticmethod
    def bottom() -> Pointer:
        return Pointer(make_graph())

    def is_bottom(self) -> bool:
        return False

    def join(self, other: Pointer) -> Pointer:
        pointers = deepcopy(self.graph)
        for obj, fields in other.graph.items():
            if obj in pointers:
                for field, values in fields.items():
                    pointers[obj][field] = pointers[obj][field] | values
            else:
                pointers[obj] = make_fields(
                    {
                        field: deepcopy(targets)
                        for field, targets in fields.items()
                        if targets
                    }
                )
        return Pointer(pointers)

    @typing.overload
    def __getitem__(self, key: Object) -> Fields:
        ...

    @typing.overload
    def __getitem__(
            self, key: tuple[Object, tac.Var]
    ) -> pythia.dom_concrete.Set[Object]:
        ...

    @typing.overload
    def __getitem__(
            self, key: tuple[pythia.dom_concrete.Set[Object], tac.Var]
    ) -> pythia.dom_concrete.Set[Object]:
        ...

    def __getitem__(self, key):
        match key:
            case Param() | Immutable() | Scope() | LocationObject() as obj:
                return self.graph[obj]
            case (
                Param() | Immutable() | Scope() | LocationObject() as obj,
                tac.Var() as var,
            ):
                return self.graph[obj][var]
            case (pythia.dom_concrete.Set() as objects, tac.Var() as var):
                return pythia.dom_concrete.Set[Object].union_all(
                    self.graph[obj][var] for obj in self.graph.keys() if obj in objects
                )
            case _:
                raise ValueError(f"Invalid key {key!r}: {type(key)}")

    @typing.overload
    def __setitem__(self, key: Object, value: Fields) -> None:
        ...

    @typing.overload
    def __setitem__(
            self, key: tuple[Object, tac.Var], value: pythia.dom_concrete.Set[Object]
    ) -> None:
        ...

    @typing.overload
    def __setitem__(
            self,
            key: tuple[pythia.dom_concrete.Set[Object], tac.Var],
            value: pythia.dom_concrete.Set[Object],
    ) -> None:
        ...

    def __setitem__(self, key, value):
        match key, value:
            case ( \
                     Param() | Immutable() | Scope() | LocationObject() as obj, \
                     tac.Var() as var, \
                 ), pythia.dom_concrete.Set() as values:
                if obj not in self.graph:
                    self.graph[obj] = make_fields({var: values})
                else:
                    self.graph[obj][var] = values
            case (
                Param() | Immutable() | Scope() | LocationObject() as obj,
                pythia.dom_concrete.Map() as fields,
            ):
                self.graph[obj] = fields
            case ( \
                     pythia.dom_concrete.Set() as objects, \
                     tac.Var() as var, \
                 ), pythia.dom_concrete.Set() as values:
                for obj in self.graph.keys():
                    if obj in objects:
                        self.graph[obj][var] = values
            case _:
                raise ValueError(f"Invalid key {key} or value {value}")

    def update(
            self, obj: Object, var: tac.Var, values: pythia.dom_concrete.Set[Object]
    ) -> None:
        if obj not in self.graph:
            self.graph[obj] = make_fields({var: values})
        else:
            self.graph[obj][var] = values

    def weak_update(
            self, obj: Object, var: tac.Var, values: pythia.dom_concrete.Set[Object]
    ) -> None:
        if obj not in self.graph:
            self.graph[obj] = make_fields({var: values})
        else:
            self.graph[obj][var] = pythia.dom_concrete.Set.join(
                self.graph[obj][var], values
            )

    def keep_keys(self, keys: typing.Iterable[Object]) -> None:
        for obj in set(self):
            if obj not in keys:
                del self.graph[obj]

    def __str__(self) -> str:
        join = (
            lambda target_obj: "{"
                               + ", ".join(sorted([str(x) for x in target_obj.as_set()]))
                               + "}"
        )
        return ", ".join(
            sorted(
                [
                    (f"{source_obj}:" if source_obj is not LOCALS else "")
                    + f"{field}->{join(target_obj)}"
                    for source_obj in self.graph
                    for field, target_obj in self.graph[source_obj].items()
                    if self.graph[source_obj][field]
                ]
            )
        )

    @staticmethod
    def initial(annotations: pythia.dom_concrete.Map[Param, ts.TypeExpr]) -> Pointer:
        return Pointer(
            make_graph(
                {
                    LOCALS: make_fields(
                        {
                            obj.param: pythia.dom_concrete.Set[Object].singleton(obj)
                            for obj in annotations
                            if obj.param is not None
                        }
                    ),
                    GLOBALS: make_fields(),
                }
            )
        )


def unop_to_str(op: tac.UnOp) -> str:
    match op:
        case tac.UnOp.BOOL:
            return "bool"
        case tac.UnOp.NEG:
            return "-"
        case tac.UnOp.NOT:
            return "not"
        case tac.UnOp.INVERT:
            return "~"
        case tac.UnOp.POS:
            return "+"
        case tac.UnOp.ITER:
            return "iter"
        case tac.UnOp.NEXT:
            return "next"
        case tac.UnOp.YIELD_ITER:
            return "yield iter"
        case _:
            raise NotImplementedError(f"UnOp.{op.name}")


def predefined(name: tac.PredefinedFunction) -> ts.TypeExpr:
    match name:
        case tac.PredefinedFunction.LIST:
            return ts.make_list_constructor()
        case tac.PredefinedFunction.SET:
            return ts.make_set_constructor()
        case tac.PredefinedFunction.TUPLE:
            return ts.make_tuple_constructor()
        case tac.PredefinedFunction.SLICE:
            return ts.make_slice_constructor()
        case tac.PredefinedFunction.MAP:
            return ts.make_dict_constructor()
    assert False, name


def build_args_typed_dict(
        arg_types: tuple[ts.TypeExpr, ...],
        kwnames: tuple[str, ...] = ()
) -> ts.TypedDict:
    """Build a TypedDict representing function arguments with keyword support.

    The last len(kwnames) args are keywords with those names.
    Positional args get Index(number, None), keyword args get Index(None, name).
    """
    n_positional = len(arg_types) - len(kwnames)
    rows = []
    # Positional: Index(number, None)
    for i, t in enumerate(arg_types[:n_positional]):
        rows.append(ts.make_row(i, None, t))
    # Keyword: Index(None, name)
    for i, name in enumerate(kwnames):
        rows.append(ts.make_row(None, name, arg_types[n_positional + i]))
    return ts.typed_dict(rows)


@dataclass
class BoundCallInfo:
    """Information stored by BoundCall for Call to retrieve."""
    applied: ts.Overloaded  # Resolved function type
    func_objects: pythia.dom_concrete.Set[Object]  # Original function
    arg_objects: tuple[pythia.dom_concrete.Set[Object], ...]  # Arguments


def retrieve_bound_call_info(
        func_obj: Object,
        func_type: ts.TypeExpr,
        prev_tp: TypedPointer,
) -> BoundCallInfo:
    """Retrieve pre-computed call info from a BoundCall object.

    Args:
        func_obj: The bound callable object (from BoundCall)
        func_type: The type stored on the bound callable (already resolved)
        prev_tp: The TypedPointer state to read from

    Returns:
        BoundCallInfo with resolved type, function objects, and arguments
    """
    # Retrieve stored arguments
    arg_objects_list: list[pythia.dom_concrete.Set[Object]] = []
    i = 0
    while True:
        arg_objs = prev_tp.pointers[func_obj, tac.Var(f"_arg{i}")]
        if not arg_objs:
            break
        arg_objects_list.append(arg_objs)
        i += 1

    # Retrieve original function objects for side effects
    func_objects = prev_tp.pointers[func_obj, tac.Var("_func")]
    if not func_objects:
        func_objects = pythia.dom_concrete.Set[Object]()

    # func_type is the already-resolved Overloaded type
    assert isinstance(func_type, ts.Overloaded), f"Expected Overloaded, got {func_type}"

    return BoundCallInfo(
        applied=func_type,
        func_objects=func_objects,
        arg_objects=tuple(arg_objects_list),
    )


def bind_method(
        location: LocationObject,
        method_type: ts.TypeExpr,
        self_objects: pythia.dom_concrete.Set[Object],
        new_tp: TypedPointer,
) -> tuple[bool, bool]:
    """Bind self to a method, creating a bound method object.

    Sets up the "self" pointer from the location to the receiver objects
    and marks the location as a bound method for explicit tracking.

    Args:
        location: The location object representing this binding site
        method_type: The type of the method being bound
        self_objects: The set of receiver objects to bind as "self"
        new_tp: The TypedPointer to update with the binding

    Returns:
        Tuple of (any_new, all_new) indicating whether binding creates new objects
    """
    if ts.is_bound_method(method_type):
        new_tp.pointers[location, tac.Var("self")] = self_objects
        new_tp.mark_bound_method(location)
        return (True, True)
    return (False, False)


def create_result_objects(
        location: LocationObject,
        applied: ts.Overloaded,
        side_effect: ts.SideEffect,
        return_type: ts.TypeExpr,
        func_objects: pythia.dom_concrete.Set[Object],
        arg_objects: tuple[pythia.dom_concrete.Set[Object], ...],
        prev_tp: TypedPointer,
        new_tp: TypedPointer,
        is_tuple_constructor: bool = False,
) -> pythia.dom_concrete.Set[Object]:
    """Create result objects based on @new, @alias, @accessor annotations.

    Handles the various ways a function call can produce result objects:
    - @new: Allocates a new object at this location
    - @alias: Creates a new object that shares fields with self
    - @accessor: Returns existing objects from self's container
    - points_to_args: The result points to the argument objects
    - Immutable: Returns a canonical immutable object

    Args:
        location: The location object representing this call site
        applied: The applied (partially resolved) function type
        side_effect: The side effect specification from the function type
        return_type: The return type of the function
        func_objects: Objects representing the function being called
        arg_objects: Objects representing the call arguments
        prev_tp: The previous TypedPointer state (read-only)
        new_tp: The TypedPointer to update with new pointers
        is_tuple_constructor: Whether this is a tuple constructor call

    Returns:
        Set of objects representing the result
    """
    pointed_objects = pythia.dom_concrete.Set[Object]()
    if side_effect.points_to_args:
        pointed_objects = pythia.dom_concrete.Set[Object].union_all(arg_objects)

    objects = pythia.dom_concrete.Set[Object]()
    creates_new = applied.any_new() or side_effect.alias

    if creates_new:
        objects = objects | pythia.dom_concrete.Set[Object].singleton(location)

        if side_effect.points_to_args:
            if is_tuple_constructor and not new_tp.pointers[location, tac.Var("*")]:
                # Tuple: use indexed fields (0, 1, 2, ...) instead of *
                for i, arg in enumerate(arg_objects):
                    new_tp.pointers[location, tac.Var(f"{i}")] = arg
            else:
                new_tp.pointers.weak_update(location, tac.Var("*"), pointed_objects)

        # Transitive @new: create objects for declared fields (only if truly new, not alias)
        if applied.any_new():
            declared_fields = ts.get_declared_fields(return_type)
            for field_name, field_type in declared_fields:
                field_obj = LocationObject((*location.location, field_name))
                new_tp.pointers[location, tac.Var(field_name)] = (
                    pythia.dom_concrete.Set[Object].singleton(field_obj)
                )
                new_tp.types[field_obj] = field_type

        # Handle @alias: copy field pointers from self to new object
        if side_effect.alias:
            func_obj = pythia.dom_concrete.Set[Object].squeeze(func_objects)
            self_objects = prev_tp.pointers[func_obj, tac.Var("self")]
            for field_name in side_effect.alias:
                field_var = tac.Var(field_name)
                for self_obj in self_objects.as_set():
                    if (self_obj, field_var) in prev_tp.pointers:
                        # Share the field pointer (aliasing!)
                        existing = prev_tp.pointers[self_obj, field_var]
                        new_tp.pointers[location, field_var] = existing

    # Handle @accessor(self[index]) - return objects from self's * field
    if side_effect.accessor:
        func_obj = pythia.dom_concrete.Set[Object].squeeze(func_objects)
        self_objects = prev_tp.pointers[func_obj, tac.Var("self")]
        star_var = tac.Var("*")
        accessor_objects = pythia.dom_concrete.Set[Object]()
        for self_obj in self_objects.as_set():
            pointed = prev_tp.pointers[self_obj, star_var]
            if pointed:
                accessor_objects = accessor_objects | pointed
        if accessor_objects:
            objects = accessor_objects
            # Type already comes from return type annotation
    elif ts.is_immutable(return_type):
        objects = immutable(return_type)
    else:
        all_new = applied.all_new() or side_effect.alias
        if not all_new:
            if ts.is_immutable(return_type):
                objects = objects | immutable(return_type)
            else:
                assert False

    return objects


def resolve_call_overload(
        func_type: ts.TypeExpr,
        arg_types: tuple[ts.TypeExpr, ...],
        kwnames: tuple[str, ...] = (),
) -> ts.Overloaded:
    """Resolve function overloads based on argument types.

    Handles constructor types (type[T]) and performs partial application
    to select matching overloads.

    Args:
        func_type: The type of the function/callable being invoked
        arg_types: Types of the positional arguments
        kwnames: Names of keyword arguments (last len(kwnames) args are keywords)

    Returns:
        The resolved Overloaded type with matching signatures
    """
    # Handle constructor calls: type[T] -> T.__init__
    if isinstance(func_type, ts.Instantiation) and func_type.generic == ts.TYPE:
        func_type = ts.get_init_func(func_type)

    assert isinstance(
        func_type, ts.Overloaded
    ), f"Expected Overloaded type, got {func_type}"

    applied = ts.partial(
        func_type, build_args_typed_dict(arg_types, kwnames), only_callable_empty=True
    )
    assert isinstance(
        applied, ts.Overloaded
    ), f"Expected Overloaded type, got {applied}"

    return applied


def apply_update_side_effects(
        side_effect: ts.SideEffect,
        func_objects: pythia.dom_concrete.Set[Object],
        arg_objects: tuple[pythia.dom_concrete.Set[Object], ...],
        prev_tp: TypedPointer,
        new_tp: TypedPointer,
) -> Dirty:
    """Apply @update side effects, modifying self's type and pointers.

    Handles the @update annotation which indicates that a method mutates
    its receiver (self). This includes:
    - Marking self and any aliased buffers as dirty
    - Updating self's type to the new type
    - Setting up element pointers from arguments to self's * field

    Args:
        side_effect: The side effect specification from the function type
        func_objects: Objects representing the function being called
        arg_objects: Objects representing the call arguments
        prev_tp: The previous TypedPointer state (read-only)
        new_tp: The TypedPointer to update with side effects

    Returns:
        Dirty map tracking which objects were modified
    """
    dirty = make_dirty()
    if side_effect.update[0] is None:
        return dirty

    func_obj = pythia.dom_concrete.Set[Object].squeeze(func_objects)
    if isinstance(func_obj, pythia.dom_concrete.Set):
        raise RuntimeError(
            f"Update with multiple function objects: {func_objects}"
        )
    self_objects = prev_tp.pointers[func_obj, tac.Var("self")]

    # Mark self objects and their buffers as dirty (for view aliasing)
    dirty_targets = self_objects
    buffer_key = tac.Var("_buffer")
    for self_obj in self_objects.as_set():
        buffer_objs = prev_tp.pointers[self_obj, buffer_key]
        if buffer_objs:  # Non-empty set
            dirty_targets = dirty_targets | buffer_objs
    dirty = make_dirty_from_keys(
        dirty_targets, pythia.dom_concrete.Set[tac.Var].top()
    )

    self_obj = pythia.dom_concrete.Set[Object].squeeze(self_objects)
    if isinstance(self_obj, pythia.dom_concrete.Set):
        raise RuntimeError(
            f"Update with multiple self objects: {self_objects}"
        )

    # Check for aliasing that would make the update unsound
    # Exclude bound methods from aliasing check - they're internal tracking objects
    aliasing_pointers = {
                            obj
                            for obj, fields in prev_tp.pointers.items()
                            for f, targets in fields.items()
                            if self_obj in targets
                            if f != tac.Var("*")  # Container element references are OK
                            if not prev_tp.is_bound_method(obj)  # Bound methods are internal
                        } - {func_obj, LOCALS}
    monomorophized = [
        obj
        for obj in aliasing_pointers
        if ts.is_monomorphized(prev_tp.types[obj])
    ]

    # NOTE: 'True or' is a debugging artifact that disables an optimization.
    # The original condition would skip updates when types already match.
    # Currently always updates (more conservative but sound). To restore
    # the optimization, remove 'True or' from the condition.
    if True or new_tp.types[self_obj] != side_effect.update[0]:
        if monomorophized:
            raise RuntimeError(
                f"Update with aliased objects: {aliasing_pointers} (not: {func_obj, LOCALS})"
            )
        new_tp.types[self_obj] = side_effect.update[0]

        # Set up element pointers from arguments
        arg_indices_to_point = side_effect.update[1]
        if arg_indices_to_point:
            for i in arg_indices_to_point:
                starred = False
                if isinstance(i, ts.Star):
                    assert len(i.items) == 1
                    i = i.items[0]
                    starred = True

                if isinstance(i, ts.Literal) and isinstance(i.value, int):
                    # TODO: minus one only for self. Should be fixed on binding
                    v = i.value - 1
                    assert v < len(arg_objects), f"{v} >= {len(arg_objects)}"
                    targets = arg_objects[v]
                    if starred:
                        targets = prev_tp.pointers[targets, tac.Var("*")]
                    new_tp.pointers.weak_update(self_obj, tac.Var("*"), targets)
                else:
                    assert False, i

    return dirty


def create_operator_result_objects(
        location: LocationObject,
        applied: ts.Overloaded,
        return_type: ts.TypeExpr,
) -> pythia.dom_concrete.Set[Object]:
    """Create result objects for operator expressions (Binary, Unary).

    Handles the common pattern of creating result objects based on:
    - Immutability of the return type
    - @new annotations on the operator method

    Args:
        location: The location object representing this operation site
        applied: The resolved operator overload
        return_type: The return type of the operator

    Returns:
        Set of objects representing the result
    """
    if ts.is_immutable(return_type):
        return immutable(return_type)

    objects = pythia.dom_concrete.Set[Object]()
    if applied.any_new():
        objects = objects | pythia.dom_concrete.Set[Object].singleton(location)
    if not applied.all_new():
        if ts.is_immutable(return_type):
            objects = objects | immutable(return_type)
        else:
            assert False, f"Non-immutable type {return_type} without @new annotation"

    return objects


@dataclass
class TypeMap:
    map: pythia.dom_concrete.Map[Object, ts.TypeExpr]

    def __repr__(self):
        return f"TypeMap({self.map})"

    def __init__(self, map: pythia.dom_concrete.Map[Object, ts.TypeExpr]):
        if not map:
            self.map = pythia.dom_concrete.Map[Object, ts.TypeExpr](map.default)
        else:
            self.map = deepcopy(map)

    def is_less_than(self, other: TypeMap) -> bool:
        # Only need to check keys in self.map because:
        # - Keys in other but not self: self[k]=BOTTOM, and is_subtype(BOTTOM, x) is always True
        # - Keys in self but not other: other[k]=BOTTOM, correctly checked below
        return all(ts.is_subtype(self.map[obj], other.map[obj]) for obj in self.map)

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return TypeMap(deepcopy(self.map, memodict))

    @staticmethod
    def bottom() -> TypeMap:
        return TypeMap(make_type_map())

    @staticmethod
    def top() -> TypeMap:
        raise NotImplementedError

    def is_bottom(self) -> bool:
        return False

    def __getitem__(self, key: Object | pythia.dom_concrete.Set[Object]) -> ts.TypeExpr:
        key = pythia.dom_concrete.Set[Object].squeeze(key)
        match key:
            case pythia.dom_concrete.Set() as objects:
                return ts.join_all(v for k, v in self.map.items() if k in objects)
            case obj:
                return self.map[obj]

    def __setitem__(
            self, key: Object | pythia.dom_concrete.Set[Object], value: ts.TypeExpr
    ) -> None:
        key = pythia.dom_concrete.Set[Object].squeeze(key)
        match key:
            case pythia.dom_concrete.Set() as objects:
                for k in self.map.keys():
                    if k in objects:
                        v = ts.join(self.map[k], value)
                        self.map[k] = v
            case k:
                self.map[k] = ts.join(self.map[k], value)

    def keep_keys(self, keys: typing.Iterable[Object]) -> None:
        for obj in set(self):
            if obj not in keys:
                del self.map[obj]

    def __iter__(self) -> typing.Iterator[Object]:
        return iter(self.map)

    def join(self, other: TypeMap) -> TypeMap:
        left = self.map
        right = other.map
        res: pythia.dom_concrete.Map[Object, ts.TypeExpr] = TypeMap.bottom().map
        for k in left.keys() | right.keys():
            res[k] = ts.join(left[k], right[k])
        return TypeMap(res)

    def __str__(self) -> str:
        return ",".join(
            sorted([f"{obj}: {type_expr}" for obj, type_expr in self.map.items()])
        )

    @staticmethod
    def initial(annotations: pythia.dom_concrete.Map[Param, ts.TypeExpr]) -> TypeMap:
        return TypeMap(annotations)


@dataclass
class TypedPointer:
    pointers: Pointer
    types: TypeMap
    dirty: Dirty
    bound_methods: BoundMethods

    def is_less_than(self: TypedPointer, other: TypedPointer) -> bool:
        # Note: bound_methods doesn't affect lattice ordering - it's metadata
        return (
                self.pointers.is_less_than(other.pointers)
                and self.types.is_less_than(other.types)
                and self.dirty.is_less_than(other.dirty)
        )

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return TypedPointer(
            deepcopy(self.pointers, memodict),
            deepcopy(self.types, memodict),
            deepcopy(self.dirty, memodict),
            deepcopy(self.bound_methods, memodict),
        )

    @staticmethod
    def bottom() -> TypedPointer:
        return TypedPointer(
            Pointer.bottom(), TypeMap.bottom(), make_dirty(), make_bound_methods()
        )

    @staticmethod
    def top() -> TypedPointer:
        raise NotImplementedError

    def is_bottom(self) -> bool:
        return self.pointers.is_bottom() or self.types.is_bottom()

    def join(self, right: TypedPointer) -> TypedPointer:
        return typed_pointer(
            self.pointers.join(right.pointers),
            self.types.join(right.types),
            self.dirty.join(right.dirty),
            self.bound_methods | right.bound_methods,
        )

    def is_bound_method(self, obj: Object) -> bool:
        """Check if an object represents a bound method."""
        return obj in self.bound_methods

    def mark_bound_method(self, obj: Object) -> None:
        """Mark an object as a bound method."""
        self.bound_methods = self.bound_methods | pythia.dom_concrete.Set[Object].singleton(obj)

    def __str__(self) -> str:
        result = (
                str(self.pointers)
                + "\n    Types: "
                + str(self.types)
                + "\n    Dirty: "
                + str(self.dirty)
        )
        if self.bound_methods:
            result += "\n    BoundMethods: " + str(self.bound_methods)
        return result

    @staticmethod
    def initial(
            annotations: pythia.dom_concrete.Map[Param, ts.TypeExpr]
    ) -> TypedPointer:
        return typed_pointer(
            Pointer.initial(annotations),
            TypeMap.initial(annotations),
            make_dirty(),
            make_bound_methods(),
        )

    def collect_garbage(self, alive: Liveness) -> None:
        if LivenessVarLattice.is_bottom(domain.Bottom):
            return

        for var in self.pointers[LOCALS].keys():
            if var.is_stackvar or True:
                if var not in alive:
                    del self.pointers[LOCALS][var]

        reachable = find_reachable_from_vars(self.pointers)
        reachable.update(obj for obj in self.types if isinstance(obj, Param))
        self.pointers.keep_keys(reachable)
        self.types.keep_keys(reachable)
        self.dirty.keep_keys(reachable)
        # Keep only reachable bound methods
        self.bound_methods = pythia.dom_concrete.Set[Object](
            obj for obj in self.bound_methods.as_set() if obj in reachable
        )

    # def normalize_types(self) -> None:
    #     # BUGGY; changes stuff that shouldn't change
    #     new_pointers = Pointer(make_graph({}))
    #     for obj, fields in self.pointers.items():
    #         for field, pointed in fields.items():
    #             new_pointers[obj, field] = frozenset(p if not isinstance(p, Immutable)
    #                                        else min(((k, v) for k, v in self.types.map.items() if v == self.types[p]),
    #                                                           key=str)[0]
    #                                                  for p in pointed)
    #     self.pointers = new_pointers


def typed_pointer(
        pointers: Pointer,
        types: TypeMap,
        dirty: Dirty,
        bound_methods: typing.Optional[BoundMethods] = None,
) -> TypedPointer:
    # Normalization.
    if pointers.is_bottom() or types.is_bottom():
        return TypedPointer.bottom()
    if bound_methods is None:
        bound_methods = make_bound_methods()
    return TypedPointer(pointers, types, dirty, bound_methods)


def parse_annotations(
        this_function: str, this_module: ts.Module
) -> pythia.dom_concrete.Map[Param, ts.TypeExpr]:
    this_signature = ts.subscr(this_module, ts.literal(this_function))
    assert isinstance(
        this_signature, ts.Overloaded
    ), f"Expected overloaded type, got {this_signature}"
    assert (
            len(this_signature.items) == 1
    ), f"Expected single signature, got {this_signature}"
    [this_signature] = this_signature.items
    annotations: dict[Param, ts.TypeExpr] = {
        Param(tac.Var(row.index.name)): row.type
        for row in this_signature.params.row_items()
        if row.index.name is not None
    }
    # annotations[Param(tac.Var('return'))] = this_signature.return_type
    return pythia.dom_concrete.Map(default=(lambda: ts.BOTTOM), d=annotations)


class TypedPointerLattice(InstructionLattice[TypedPointer]):
    liveness: dict[Location, Liveness]
    annotations: pythia.dom_concrete.Map[Param, ts.TypeExpr]
    backward: bool = False
    for_locations: frozenset[Location]

    @classmethod
    def name(cls) -> str:
        return "TypedPointer"

    def __init__(
            self,
            liveness: dict[Location, MapDomain[tac.Var, Liveness]],
            this_function: str,
            this_module: ts.Module,
            for_locations: frozenset[Location],
    ) -> None:
        super().__init__()
        self.annotations = parse_annotations(this_function, this_module)
        self.this_module = this_module
        self.builtins = ts.resolve_static_ref(ts.Ref("builtins"))
        self.liveness = liveness
        self.for_locations = for_locations

    def is_less_than(self, left: TypedPointer, right: TypedPointer) -> bool:
        return left.is_less_than(right)

    def initial(self) -> TypedPointer:
        return TypedPointer.initial(self.annotations)

    def bottom(self) -> TypedPointer:
        return TypedPointer.bottom()

    def top(self) -> TypedPointer:
        raise NotImplementedError

    def is_bottom(self, tp: TypedPointer) -> bool:
        return tp.is_bottom()

    def join(self, left: TypedPointer, right: TypedPointer) -> TypedPointer:
        return left.join(right)

    def resolve(self, ref: ts.TypeExpr) -> ts.TypeExpr:
        if isinstance(ref, ts.Ref):
            if "." in ref.name:
                module, name = ref.name.split(".", 1)
                if module == self.this_module.name:
                    return ts.subscr(self.this_module, ts.literal(name))
        return ref

    def attribute(self, t: ts.TypeExpr, attr: tac.Var) -> ts.TypeExpr:
        mod = self.resolve(t)
        assert mod != ts.TOP, f"Cannot resolve {attr} in {t}"
        try:
            # FIX: How to differentiate nonexistent attributes from attributes that are TOP?
            res = ts.subscr(mod, ts.literal(attr.name))
            # Fix: only works for depth 1
            if isinstance(mod, ts.Module):
                mod = ts.Ref(mod.name)
            match mod, res:
                case ts.Ref(name=modname), ts.Instantiation(
                    ts.Ref("builtins.type"), (ts.Class(), )
                ):
                    arg = ts.Ref(f"{modname}.{attr.name}")
                    return replace(res, type_args=(arg,))
            if res == ts.BOTTOM:
                if mod == self.this_module:
                    return ts.subscr(self.builtins, ts.literal(attr.name))
            return res
        except TypeError:
            if mod == self.this_module:
                return ts.subscr(self.builtins, ts.literal(attr.name))
            raise

    def _expr_const(
            self,
            expr: tac.Const,
    ) -> tuple[pythia.dom_concrete.Set[Object], ts.TypeExpr, Dirty]:
        """Handle constant expressions."""
        t = ts.literal(expr.value)
        return (immutable(t), t, make_dirty())

    def _expr_var(
            self,
            prev_tp: TypedPointer,
            expr: tac.Var,
    ) -> tuple[pythia.dom_concrete.Set[Object], ts.TypeExpr, Dirty]:
        """Handle variable expressions."""
        objs = prev_tp.pointers[LOCALS, expr]
        types = prev_tp.types[objs]
        return (objs, types, make_dirty())

    def _expr_attribute_globals(
            self,
            prev_tp: TypedPointer,
            field: tac.Var,
    ) -> tuple[pythia.dom_concrete.Set[Object], ts.TypeExpr, Dirty]:
        """Handle global attribute access."""
        global_objs = prev_tp.pointers[GLOBALS, field]
        assert not global_objs
        t = ts.subscr(self.this_module, ts.literal(field.name))
        ref = None
        if t == ts.BOTTOM:
            ref = ts.Ref(f"builtins.{field}")
        if isinstance(t, ts.Ref):
            ref = t
        if ref is not None:
            t = ts.resolve_static_ref(ref)
        match t:
            case ts.Instantiation(ts.Ref("builtins.type"), (ts.Class(), )) as t:
                t = replace(t, type_args=((ref,)))
            case ts.Class():
                t = ts.get_return(t)
        # TODO: class through type
        return (immutable(t), t, make_dirty())

    def _expr_attribute_var(
            self,
            prev_tp: TypedPointer,
            var: tac.Var,
            field: tac.Var,
            location: LocationObject,
            new_tp: TypedPointer,
    ) -> tuple[pythia.dom_concrete.Set[Object], ts.TypeExpr, Dirty]:
        """Handle attribute access on a variable."""
        var_objs = prev_tp.pointers[LOCALS, var]
        attr_type = self.attribute(prev_tp.types[var_objs], field)
        any_new = all_new = False
        side_effect = ts.SideEffect(new=False)
        assert not isinstance(attr_type, ts.FunctionType)
        if isinstance(attr_type, ts.Overloaded):
            if any(item.is_property for item in attr_type.items):
                assert all(f.is_property for f in attr_type.items)
                any_new = attr_type.any_new()
                all_new = attr_type.all_new()
                side_effect = ts.get_side_effect(attr_type)
                attr_type = ts.get_return(attr_type)
            bound_any, bound_all = bind_method(
                location, attr_type, var_objs, new_tp
            )
            any_new = any_new or bound_any
            all_new = all_new or bound_all
        t = attr_type

        # @alias creates a new object with shared fields
        creates_new = any_new or side_effect.alias
        all_creates_new = all_new or side_effect.alias

        if ts.is_immutable(t):
            objects = immutable(t)
        else:
            objects = prev_tp.pointers[var_objs, field]
            if creates_new:
                objects = objects | pythia.dom_concrete.Set[Object].singleton(
                    location
                )
                # Handle @alias: copy field pointers from self to new object
                if side_effect.alias:
                    for field_name in side_effect.alias:
                        field_var = tac.Var(field_name)
                        for var_obj in var_objs.as_set():
                            existing = prev_tp.pointers[var_obj, field_var]
                            if existing:  # Non-empty set
                                # Share the field pointer (aliasing!)
                                new_tp.pointers[location, field_var] = existing
            if not all_creates_new:
                if ts.is_immutable(t):
                    objects = objects | immutable(t)
                else:
                    assert False

        return (objects, t, make_dirty())

    def _expr_subscript(
            self,
            prev_tp: TypedPointer,
            var: tac.Var,
            index: tac.Var,
            location: LocationObject,
    ) -> tuple[pythia.dom_concrete.Set[Object], ts.TypeExpr, Dirty]:
        """Handle subscript expressions."""
        var_objs = prev_tp.pointers[LOCALS, var]
        index_objs = prev_tp.pointers[LOCALS, index]
        index_type = self.resolve(prev_tp.types[index_objs])
        selftype = self.resolve(prev_tp.types[var_objs])
        t = ts.subscr(selftype, index_type)
        any_new = all_new = False
        is_accessor = False
        if isinstance(t, ts.Overloaded) and any(
                item.is_property for item in t.items
        ):
            assert all(f.is_property for f in t.items)
            any_new = t.any_new()
            all_new = t.all_new()
            # Check if __getitem__ has @accessor annotation
            is_accessor = any(
                item.side_effect.accessor for item in t.items
            )
            t = ts.get_return(t)
        assert t != ts.BOTTOM, f"Subscript {var}[{index_type}] is BOTTOM"
        direct_objs = prev_tp.pointers[var_objs, tac.Var("*")]
        # TODO: class through type

        # Handle @accessor: return objects from container's * field
        if is_accessor and direct_objs:
            objects = direct_objs
        elif ts.is_immutable(t):
            objects = immutable(t)
        else:
            objects = direct_objs
            if any_new:
                if all_new:
                    # TODO: assert not direct_objs ??
                    objects = pythia.dom_concrete.Set[Object].singleton(
                        location
                    )
                else:
                    objects = objects | pythia.dom_concrete.Set[
                        Object
                    ].singleton(location)
            if not all_new:
                if ts.is_immutable(t):
                    objects = objects | immutable(t)

        return (objects, t, make_dirty())

    def _expr_bound_call(
            self,
            prev_tp: TypedPointer,
            var: tac.Var | tac.PredefinedFunction,
            args: tuple[tac.Var, ...],
            kwnames: tuple[str, ...],
            location: LocationObject,
            new_tp: TypedPointer,
    ) -> tuple[pythia.dom_concrete.Set[Object], ts.TypeExpr, Dirty]:
        """Handle BoundCall expressions - prepare a callable for later execution.

        BoundCall performs the "pure" preparation phase:
        - Function/method lookup
        - Argument collection
        - Overload resolution

        Stores on the bound callable object (location):
        - _arg0, _arg1, ... : Argument object sets
        - _func : Original function objects (for side effects)
        - self : Receiver object (if method call)
        - Type slot : The resolved Overloaded type

        Call retrieves this info via retrieve_bound_call_info() and:
        - Applies side effects (mutations to receiver/args)
        - Creates result objects (with Call's location)
        - Computes dirty tracking

        This separation matches the paper's Bind/Call distinction.
        """
        # Step 1: Function lookup
        if isinstance(var, tac.Var):
            func_objects = prev_tp.pointers[LOCALS, var]
            func_type = prev_tp.types[func_objects]
        elif isinstance(var, tac.PredefinedFunction):
            func_objects = pythia.dom_concrete.Set[Object]()
            func_type = predefined(var)
        else:
            assert False, f"Expected Var or PredefinedFunction, got {var}"

        # Step 2: Argument collection
        arg_objects = tuple([prev_tp.pointers[LOCALS, v] for v in args])
        arg_types = tuple([prev_tp.types[obj] for obj in arg_objects])
        assert all(
            arg for arg in arg_objects
        ), f"Expected non-empty arg objects, got {arg_objects}"

        # Step 3: Overload resolution
        applied = resolve_call_overload(func_type, arg_types, kwnames)

        # Step 4: Create bound callable object
        bound_objects = pythia.dom_concrete.Set[Object].singleton(location)

        # Copy self pointer if function is a bound method
        if func_objects:
            func_obj = pythia.dom_concrete.Set[Object].squeeze(func_objects)
            if not isinstance(func_obj, pythia.dom_concrete.Set):
                self_objs = prev_tp.pointers[func_obj, tac.Var("self")]
                if self_objs:
                    new_tp.pointers[location, tac.Var("self")] = self_objs

        # Store arg objects for Call to retrieve
        for i, arg_obj in enumerate(arg_objects):
            new_tp.pointers[location, tac.Var(f"_arg{i}")] = arg_obj

        # Store the original function objects for side effect handling
        if func_objects:
            new_tp.pointers[location, tac.Var("_func")] = func_objects

        # Mark as bound callable
        new_tp.mark_bound_method(location)

        return (bound_objects, applied, make_dirty())

    def _expr_call(
            self,
            prev_tp: TypedPointer,
            var: tac.Var | tac.PredefinedFunction,
            args: tuple[tac.Var, ...],
            kwnames: tuple[str, ...],
            location: LocationObject,
            new_tp: TypedPointer,
    ) -> tuple[pythia.dom_concrete.Set[Object], ts.TypeExpr, Dirty]:
        """Handle Call expressions - execute a callable and produce results."""
        if isinstance(var, tac.Var):
            func_objects = prev_tp.pointers[LOCALS, var]
            func_type = prev_tp.types[func_objects]

            # Check if calling a bound callable (from BoundCall)
            func_obj = pythia.dom_concrete.Set[Object].squeeze(func_objects)
            is_bound_call = (
                    not isinstance(func_obj, pythia.dom_concrete.Set)
                    and prev_tp.is_bound_method(func_obj)
                    and not args  # Bound calls have empty args
            )

            if is_bound_call:
                # Use helper to retrieve pre-computed info
                info = retrieve_bound_call_info(func_obj, func_type, prev_tp)
                applied = info.applied
                arg_objects = info.arg_objects
                func_objects = info.func_objects
            else:
                # Direct call path (legacy or LIST_APPEND pattern)
                arg_objects = tuple(
                    [prev_tp.pointers[LOCALS, v] for v in args]
                )
                arg_types = tuple([prev_tp.types[obj] for obj in arg_objects])
                assert all(
                    arg for arg in arg_objects
                ), f"Expected non-empty arg objects, got {arg_objects}"
                applied = resolve_call_overload(func_type, arg_types, kwnames)
        elif isinstance(var, tac.PredefinedFunction):
            # TODO: point from exact literal when possible
            func_objects = pythia.dom_concrete.Set[Object]()
            func_type = predefined(var)
            arg_objects = tuple([prev_tp.pointers[LOCALS, v] for v in args])
            arg_types = tuple([prev_tp.types[obj] for obj in arg_objects])
            assert all(
                arg for arg in arg_objects
            ), f"Expected non-empty arg objects, got {arg_objects}"
            applied = resolve_call_overload(func_type, arg_types, kwnames)
        else:
            assert False, f"Expected Var or PredefinedFunction, got {var}"

        side_effect = ts.get_side_effect(applied)
        dirty = apply_update_side_effects(
            side_effect, func_objects, arg_objects, prev_tp, new_tp
        )

        t = ts.get_return(applied)
        assert t != ts.BOTTOM, f"Expected non-bottom return type for {locals()}"

        objects = create_result_objects(
            location,
            applied,
            side_effect,
            t,
            func_objects,
            arg_objects,
            prev_tp,
            new_tp,
            is_tuple_constructor=(var == tac.PredefinedFunction.TUPLE),
        )

        assert objects
        return (objects, t, dirty)

    def _expr_unary(
            self,
            prev_tp: TypedPointer,
            var: tac.Var,
            op: tac.UnOp,
            location: LocationObject,
    ) -> tuple[pythia.dom_concrete.Set[Object], ts.TypeExpr, Dirty]:
        """Handle unary expressions."""
        value_objects = prev_tp.pointers[LOCALS, var]
        assert value_objects, f"Expected objects for {var}"
        arg_type = prev_tp.types[value_objects]
        applied = ts.get_unop(arg_type, unop_to_str(op))
        assert isinstance(
            applied, ts.Overloaded
        ), f"Expected overloaded type, got {applied}"

        side_effect = ts.get_side_effect(applied)
        dirty = make_dirty()
        if side_effect.update[0] is not None:
            dirty = make_dirty_from_keys(
                value_objects, pythia.dom_concrete.Set[tac.Var].top()
            )

        t = ts.get_return(applied)
        objects = create_operator_result_objects(location, applied, t)

        return (objects, t, dirty)

    def _expr_binary(
            self,
            prev_tp: TypedPointer,
            left: tac.Var,
            right: tac.Var,
            op: str,
            location: LocationObject,
    ) -> tuple[pythia.dom_concrete.Set[Object], ts.TypeExpr, Dirty]:
        """Handle binary expressions."""
        left_objects = prev_tp.pointers[LOCALS, left]
        right_objects = prev_tp.pointers[LOCALS, right]
        left_type = prev_tp.types[left_objects]
        right_type = prev_tp.types[right_objects]
        applied = ts.partial_binop(left_type, right_type, op)
        assert isinstance(
            applied, ts.Overloaded
        ), f"Expected overloaded type, got {applied}"

        t = ts.get_return(applied)
        objects = create_operator_result_objects(location, applied, t)

        return (objects, t, make_dirty())

    def expr(
            self,
            prev_tp: TypedPointer,
            expr: tac.Expr,
            location: LocationObject,
            new_tp: TypedPointer,
    ) -> tuple[pythia.dom_concrete.Set[Object], ts.TypeExpr, Dirty]:
        """Dispatch to the appropriate expression handler based on expression type."""
        match expr:
            case tac.Const():
                return self._expr_const(expr)
            case tac.Var():
                return self._expr_var(prev_tp, expr)
            case tac.Attribute(var=tac.PredefinedScope.GLOBALS, field=tac.Var() as field):
                return self._expr_attribute_globals(prev_tp, field)
            case tac.Attribute(var=tac.Var() as var, field=tac.Var() as field):
                return self._expr_attribute_var(prev_tp, var, field, location, new_tp)
            case tac.Subscript(var=tac.Var() as var, index=tac.Var() as index):
                return self._expr_subscript(prev_tp, var, index, location)
            case tac.BoundCall(var, tuple() as args, tuple() as kwnames):
                return self._expr_bound_call(prev_tp, var, args, kwnames, location, new_tp)
            case tac.Call(var, tuple() as args, tuple() as kwnames):
                return self._expr_call(prev_tp, var, args, kwnames, location, new_tp)
            case tac.Unary(var=tac.Var() as var, op=tac.UnOp() as op):
                return self._expr_unary(prev_tp, var, op, location)
            case tac.Binary(left=tac.Var() as left, right=tac.Var() as right, op=str() as op):
                return self._expr_binary(prev_tp, left, right, op, location)
            case _:
                raise NotImplementedError(expr)

    def signature(
            self,
            tp: TypedPointer,
            signature: tac.Signature,
            pointed: pythia.dom_concrete.Set[Object],
            t: ts.TypeExpr,
    ) -> None:
        match signature:
            case None:
                pass
            case tuple() as signature:  # type: ignore
                unknown = tp.pointers[pointed, tac.Var("*")]
                indirect_pointed = [
                    tp.pointers[pointed, tac.Var(f"{i}")] | unknown
                    for i in range(len(signature))
                ]
                for i, var in enumerate(signature):
                    assert isinstance(var, tac.Var), f"Expected Var, got {var}"
                    objs = indirect_pointed[i]
                    ti = ts.subscr(t, ts.literal(i))
                    if isinstance(ti, ts.Overloaded):
                        assert all(
                            f.is_property for f in ti.items
                        ), f"Expected all properties, got {ti}"
                        ti = ts.get_return(ti)
                    if objs:
                        tp.pointers[LOCALS, var] = objs
                    else:
                        assert ts.is_immutable(ti), f"Expected immutable type, got {ti}"
                        objs = immutable(ti)
                    tp.pointers[LOCALS, var] = objs
                    tp.types[objs] = ti
            case tac.Var() as var:
                tp.pointers[LOCALS, var] = pointed
                tp.types[pointed] = t
                new_dirty = make_dirty_from_keys(
                    pythia.dom_concrete.Set[Object].singleton(LOCALS),
                    pythia.dom_concrete.Set[tac.Var].singleton(var),
                )
                tp.dirty = tp.dirty.join(new_dirty)
            case tac.Attribute(var=var, field=field):
                targets = tp.pointers[LOCALS, var]
                tp.pointers[targets, field] = pointed
                new_dirty = make_dirty_from_keys(
                    targets, pythia.dom_concrete.Set[tac.Var].singleton(field)
                )
                tp.dirty = tp.dirty.join(new_dirty)
            case tac.Subscript(var=var):
                targets = tp.pointers[LOCALS, var]
                tp.pointers[targets, tac.Var("*")] = pointed
                new_dirty = make_dirty_from_keys(
                    targets,
                    pythia.dom_concrete.Set[tac.Var].singleton(tac.Var("*")),
                )
                tp.dirty = tp.dirty.join(new_dirty)
            case _:
                assert False, f"unexpected signature {signature}"

    def transfer(
            self, prev_tp: TypedPointer, ins: tac.Tac, location: Location
    ) -> TypedPointer:
        tp = deepcopy(prev_tp)

        if location in self.for_locations:
            tp.dirty = make_dirty()

        if isinstance(ins, tac.For):
            ins = ins.as_call()

        # FIX: this removes pointers and make it "bottom" instead of "top"
        for var in tac.gens(ins):
            if var in tp.pointers[LOCALS]:
                del tp.pointers[LOCALS][var]

        match ins:
            case tac.Assign(lhs, expr, or_null):
                (pointed, types, new_dirty) = self.expr(
                    prev_tp, expr, LocationObject(location), tp
                )
                if types == ts.BOTTOM:
                    pass
                if or_null and len(pointed.as_set()) == 0:
                    t = ts.literal(ts.NULL)
                    (pointed, types, new_dirty) = (immutable(t), t, make_dirty())
                tp.dirty = tp.dirty.join(new_dirty)
                self.signature(tp, lhs, pointed, types)
            case tac.Return(var):
                val = tp.pointers[LOCALS, var]
                tp.pointers[LOCALS, tac.Var("return")] = val

        # tp.normalize_types()
        tp.collect_garbage(self.liveness[location])

        # assert old_prev_tp == prev_tp, f'{old_prev_tp}\n{prev_tp}'
        return tp


# Kept for debugging - call with print_debug(ins, tp) to see variable states
# def print_debug(ins: tac.Tac, tp: TypedPointer) -> None:
#     for var in tac.free_vars(ins):
#         if var in tp.pointers[LOCALS].keys():
#             p = tp.pointers[LOCALS, var]
#             t = tp.types[p]
#             print(f"  {var} = {p} : {t}")
#         else:
#             print(f"  {var} = <bottom>")


def find_reachable_from_vars(ptr: Pointer) -> set[Object]:
    worklist = {LOCALS}
    reachable = set(worklist)
    while worklist:
        root = worklist.pop()
        for edge, objects in ptr[root].items():
            object_set = objects.as_set()
            worklist.update(object_set - reachable)
            reachable.update(object_set)
    return reachable


def find_reachable(
        ptr: Pointer,
        alive: set[tac.Var],
        params: set[tac.Var],
        sources: typing.Optional[typing.Iterable[Object]] = None,
) -> typing.Iterator[LocationObject]:
    worklist = set(sources) if sources is not None else {LOCALS}
    while worklist:
        root = worklist.pop()
        if isinstance(root, LocationObject):
            yield root
        for edge, objects in ptr[root].items():
            if root == LOCALS and edge not in alive:
                # We did not remove non-stack variables from the pointer lattice, so we need to filter them out here.
                continue
            if edge in params:
                continue
            for obj in objects.as_set():
                if isinstance(obj, Param):
                    continue
                worklist.add(obj)


def find_dirty_roots(tp: TypedPointer, liveness: Liveness) -> typing.Iterator[str]:
    assert not isinstance(liveness, domain.Bottom)
    alive = liveness.as_set()
    for var in tp.dirty[LOCALS].as_set():
        if var.is_stackvar:
            continue
        if not var in alive:
            continue
        yield var.name
    for var, target in tp.pointers[LOCALS].items():
        if var.is_stackvar:
            continue
        if var.name == "return":
            continue
        if not var in alive:
            continue
        reachable = set(find_reachable(tp.pointers, alive, set(), target.as_set()))
        if any(tp.dirty[obj] for obj in reachable):
            yield var.name
