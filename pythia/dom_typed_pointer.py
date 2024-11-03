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
# 1. line of code where it is allocated
# 2. parameter
# 3. immutable value of a certain type
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
        return f"@location {self.location}"


type Object = typing.Union[LocationObject, Param, Immutable, Scope]

type Fields = pythia.dom_concrete.Map[tac.Var, pythia.dom_concrete.Set[Object]]
type Graph = pythia.dom_concrete.Map[Object, Fields]

type Dirty = pythia.dom_concrete.Map[Object, pythia.dom_concrete.Set[tac.Var]]


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
        return all(
            self.graph[obj][field].is_subset(other.graph[obj][field])
            for obj in self.graph
            for field in self.graph[obj]
        )

    def __deepcopy__(self, memodict={}):
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
    def __getitem__(self, key: Object) -> Fields: ...

    @typing.overload
    def __getitem__(
        self, key: tuple[Object, tac.Var]
    ) -> pythia.dom_concrete.Set[Object]: ...

    @typing.overload
    def __getitem__(
        self, key: tuple[pythia.dom_concrete.Set[Object], tac.Var]
    ) -> pythia.dom_concrete.Set[Object]: ...

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
    def __setitem__(self, key: Object, value: Fields) -> None: ...

    @typing.overload
    def __setitem__(
        self, key: tuple[Object, tac.Var], value: pythia.dom_concrete.Set[Object]
    ) -> None: ...

    @typing.overload
    def __setitem__(
        self,
        key: tuple[pythia.dom_concrete.Set[Object], tac.Var],
        value: pythia.dom_concrete.Set[Object],
    ) -> None: ...

    def __setitem__(self, key, value):
        match key, value:
            case (
                Param() | Immutable() | Scope() | LocationObject() as obj,
                tac.Var() as var,
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
            case (
                pythia.dom_concrete.Set() as objects,
                tac.Var() as var,
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

    def keep_keys(self, keys: typing.Iterable[Object]) -> None:
        for obj in set(self):
            if obj not in keys:
                del self.graph[obj]

    def __str__(self) -> str:
        join = (
            lambda target_obj: "{"
            + ", ".join(str(x) for x in target_obj.as_set())
            + "}"
        )
        return ", ".join(
            (f"{source_obj}:" if source_obj is not LOCALS else "")
            + f"{field}->{join(target_obj)}"
            for source_obj in self.graph
            for field, target_obj in self.graph[source_obj].items()
            if self.graph[source_obj][field]
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
    assert False, name


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
        # TODO: check
        return all(ts.is_subtype(self.map[obj], other.map[obj]) for obj in self.map)

    def __deepcopy__(self, memodict={}):
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
                # Weak update; not a singleton
                for k in self.map.keys():
                    if k in objects:
                        v = ts.join(self.map[k], value)
                        self.map[k] = v
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
        res: pythia.dom_concrete.Map[Object, ts.TypeExpr] = TypeMap.bottom().map
        for k in left.keys() | right.keys():
            res[k] = ts.join(left[k], right[k])
        return TypeMap(res)

    def __str__(self) -> str:
        return ",".join(f"{obj}: {type_expr}" for obj, type_expr in self.map.items())

    @staticmethod
    def initial(annotations: pythia.dom_concrete.Map[Param, ts.TypeExpr]) -> TypeMap:
        return TypeMap(annotations)


@dataclass
class TypedPointer:
    pointers: Pointer
    types: TypeMap
    dirty: Dirty

    def is_less_than(self: TypedPointer, other: TypedPointer) -> bool:
        return (
            self.pointers.is_less_than(other.pointers)
            and self.types.is_less_than(other.types)
            and self.dirty.is_less_than(other.dirty)
        )

    def __deepcopy__(self, memodict={}):
        return TypedPointer(
            deepcopy(self.pointers, memodict),
            deepcopy(self.types, memodict),
            deepcopy(self.dirty, memodict),
        )

    @staticmethod
    def bottom() -> TypedPointer:
        return TypedPointer(Pointer.bottom(), TypeMap.bottom(), make_dirty())

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
        )

    def __str__(self) -> str:
        return str(self.pointers) + "\n" + str(self.types)

    @staticmethod
    def initial(
        annotations: pythia.dom_concrete.Map[Param, ts.TypeExpr]
    ) -> TypedPointer:
        return typed_pointer(
            Pointer.initial(annotations), TypeMap.initial(annotations), make_dirty()
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


def typed_pointer(pointers: Pointer, types: TypeMap, dirty: Dirty) -> TypedPointer:
    # Normalization.
    if pointers.is_bottom() or types.is_bottom():
        return TypedPointer.bottom()
    return TypedPointer(pointers, types, dirty)


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
                    ts.Ref("builtins.type"), (ts.Class(),)
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

    def expr(
        self,
        prev_tp: TypedPointer,
        expr: tac.Expr,
        location: LocationObject,
        new_tp: TypedPointer,
    ) -> tuple[pythia.dom_concrete.Set[Object], ts.TypeExpr, Dirty]:
        objects: pythia.dom_concrete.Set[Object]
        dirty: Dirty
        match expr:
            case tac.Const(value):
                t = ts.literal(value)
                return (immutable(t), t, make_dirty())
            case tac.Var() as var:
                objs = prev_tp.pointers[LOCALS, var]
                types = prev_tp.types[objs]
                return (objs, types, make_dirty())
            case tac.Attribute(
                var=tac.PredefinedScope.GLOBALS, field=tac.Var() as field
            ):
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
                    case ts.Instantiation(ts.Ref("builtins.type"), (ts.Class(),)) as t:
                        t = replace(t, type_args=((ref,)))
                    case ts.Class():
                        t = ts.get_return(t)
                # TODO: class through type
                return (immutable(t), t, make_dirty())
            case tac.Attribute(var=tac.Var() as var, field=tac.Var() as field):
                var_objs = prev_tp.pointers[LOCALS, var]
                t = self.attribute(prev_tp.types[var_objs], field)
                any_new = all_new = False
                assert not isinstance(t, ts.FunctionType)
                if isinstance(t, ts.Overloaded):
                    if any(item.is_property for item in t.items):
                        assert all(f.is_property for f in t.items)
                        any_new = t.any_new()
                        all_new = t.all_new()
                        t = ts.get_return(t)
                    if ts.is_bound_method(t):
                        new_tp.pointers[location, tac.Var("self")] = var_objs
                        any_new = True
                        all_new = True

                if ts.is_immutable(t):
                    objects = immutable(t)
                else:
                    objects = prev_tp.pointers[var_objs, field]
                    if any_new:
                        objects = objects | pythia.dom_concrete.Set[Object].singleton(
                            location
                        )
                    if not all_new:
                        if ts.is_immutable(t):
                            objects = objects | immutable(t)
                        else:
                            assert False

                return (objects, t, make_dirty())
            case tac.Subscript(var=tac.Var() as var, index=tac.Var() as index):
                var_objs = prev_tp.pointers[LOCALS, var]
                index_objs = prev_tp.pointers[LOCALS, index]
                index_type = prev_tp.types[index_objs]
                var_type = prev_tp.types[var_objs]
                selftype = self.resolve(var_type)
                index = self.resolve(index_type)
                t = ts.subscr(selftype, index)
                any_new = all_new = False
                if isinstance(t, ts.Overloaded) and any(
                    item.is_property for item in t.items
                ):
                    assert all(f.is_property for f in t.items)
                    any_new = t.any_new()
                    all_new = t.all_new()
                    t = ts.get_return(t)
                assert t != ts.BOTTOM, f"Subscript {var}[{index}] is BOTTOM"
                direct_objs = prev_tp.pointers[var_objs, tac.Var("*")]
                # TODO: class through type

                if ts.is_immutable(t):
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
            case tac.Call(var, tuple() as args):
                if isinstance(var, tac.Var):
                    func_objects = prev_tp.pointers[LOCALS, var]
                    func_type = prev_tp.types[func_objects]
                elif isinstance(var, tac.PredefinedFunction):
                    # TODO: point from exact literal when possible
                    func_objects = pythia.dom_concrete.Set[Object]()
                    func_type = predefined(var)
                else:
                    assert False, f"Expected Var or PredefinedFunction, got {var}"
                if isinstance(
                    func_type, ts.Instantiation
                ) and func_type.generic == ts.Ref("builtins.type"):
                    func_type = ts.get_init_func(func_type)
                assert isinstance(
                    func_type, ts.Overloaded
                ), f"Expected Overloaded type, got {func_type}"
                arg_objects = tuple([prev_tp.pointers[LOCALS, var] for var in args])
                arg_types = tuple([prev_tp.types[obj] for obj in arg_objects])
                assert all(
                    arg for arg in arg_objects
                ), f"Expected non-empty arg objects, got {arg_objects}"
                applied = ts.partial(
                    func_type, ts.positional(*arg_types), only_callable_empty=True
                )
                assert isinstance(
                    applied, ts.Overloaded
                ), f"Expected Overloaded type, got {applied}"

                side_effect = ts.get_side_effect(applied)
                dirty = make_dirty()
                if side_effect.update is not None:
                    func_obj = pythia.dom_concrete.Set[Object].squeeze(func_objects)
                    if isinstance(func_obj, pythia.dom_concrete.Set):
                        raise RuntimeError(
                            f"Update with multiple function objects: {func_objects}"
                        )
                    self_objects = prev_tp.pointers[func_obj, tac.Var("self")]
                    dirty = make_dirty_from_keys(
                        self_objects, pythia.dom_concrete.Set[tac.Var].top()
                    )
                    self_obj = pythia.dom_concrete.Set[Object].squeeze(self_objects)
                    if isinstance(self_obj, pythia.dom_concrete.Set):
                        raise RuntimeError(
                            f"Update with multiple self objects: {self_objects}"
                        )

                    aliasing_pointers = {
                        obj
                        for obj, fields in prev_tp.pointers.items()
                        for f, targets in fields.items()
                        if self_obj in targets
                    } - {func_obj, LOCALS}
                    monomorophized = [
                        obj
                        for obj in aliasing_pointers
                        if ts.is_monomorphized(prev_tp.types[obj])
                    ]
                    # Expected two objects: self argument and locals

                    if new_tp.types[self_obj] != side_effect.update:
                        if monomorophized:
                            raise RuntimeError(
                                f"Update with aliased objects: {aliasing_pointers} (not: {func_obj, LOCALS})"
                            )
                        new_tp.types[self_obj] = side_effect.update
                        if side_effect.name == "append":
                            x = arg_objects[0]
                            new_tp.pointers.update(self_obj, tac.Var("*"), x)

                t = ts.get_return(applied)
                assert t != ts.BOTTOM, f"Expected non-bottom return type for {locals()}"

                pointed_objects = pythia.dom_concrete.Set[Object]()
                if side_effect.points_to_args:
                    pointed_objects = pythia.dom_concrete.Set[Object].union_all(
                        arg_objects
                    )

                objects = pythia.dom_concrete.Set[Object]()
                if applied.any_new():
                    objects = objects | pythia.dom_concrete.Set[Object].singleton(
                        location
                    )
                    if side_effect.points_to_args:
                        if (
                            var == tac.PredefinedFunction.TUPLE
                            and not new_tp.pointers[location, tac.Var("*")]
                        ):
                            for i, arg in enumerate(arg_objects):
                                new_tp.pointers[location, tac.Var(f"{i}")] = arg
                        else:
                            new_tp.pointers.update(
                                location, tac.Var("*"), pointed_objects
                            )

                if ts.is_immutable(t):
                    objects = immutable(t)
                else:
                    if not applied.all_new():
                        if ts.is_immutable(t):
                            objects = objects | immutable(t)
                        else:
                            assert False
                    else:
                        pass
                        # # TODO: actually "returns args"
                        # if side_effect.points_to_args:
                        #     objects = objects | pointed_objects

                assert objects
                return (objects, t, dirty)
            case tac.Unary(var=tac.Var() as var, op=tac.UnOp() as op):
                value_objects = prev_tp.pointers[LOCALS, var]
                assert value_objects, f"Expected objects for {var}"
                arg_type = prev_tp.types[value_objects]
                applied = ts.get_unop(arg_type, unop_to_str(op))
                assert isinstance(applied, ts.Overloaded)
                side_effect = ts.get_side_effect(applied)
                dirty = make_dirty()
                if side_effect.update is not None:
                    dirty = make_dirty_from_keys(
                        value_objects, pythia.dom_concrete.Set[tac.Var].top()
                    )
                assert isinstance(
                    applied, ts.Overloaded
                ), f"Expected overloaded type, got {applied}"

                t = ts.get_return(applied)

                if ts.is_immutable(t):
                    objects = immutable(t)
                else:
                    objects = pythia.dom_concrete.Set[Object]()
                    if applied.any_new():
                        objects = objects | pythia.dom_concrete.Set[Object].singleton(
                            location
                        )
                    if not applied.all_new():
                        if ts.is_immutable(t):
                            objects = objects | immutable(t)
                        else:
                            assert False

                return (objects, t, dirty)
            case tac.Binary(
                left=tac.Var() as left, right=tac.Var() as right, op=str() as op
            ):
                left_objects = prev_tp.pointers[LOCALS, left]
                right_objects = prev_tp.pointers[LOCALS, right]
                left_type = prev_tp.types[left_objects]
                right_type = prev_tp.types[right_objects]
                applied = ts.partial_binop(left_type, right_type, op)
                assert isinstance(
                    applied, ts.Overloaded
                ), f"Expected overloaded type, got {applied}"

                t = ts.get_return(applied)

                if ts.is_immutable(t):
                    objects = immutable(t)
                else:
                    objects = pythia.dom_concrete.Set[Object]()
                    if applied.any_new():
                        objects = objects | pythia.dom_concrete.Set[Object].singleton(
                            location
                        )
                    if not applied.all_new():
                        if ts.is_immutable(t):
                            objects = objects | immutable(t)
                        else:
                            assert False

                return (objects, t, make_dirty())
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
                tp.dirty.update(
                    make_dirty_from_keys(
                        pythia.dom_concrete.Set[Object].singleton(LOCALS),
                        pythia.dom_concrete.Set[tac.Var].singleton(var),
                    )
                )
            case tac.Attribute(var=var, field=field):
                targets = tp.pointers[LOCALS, var]
                tp.pointers[targets, field] = pointed
                tp.dirty.update(
                    make_dirty_from_keys(
                        targets, pythia.dom_concrete.Set[tac.Var].singleton(field)
                    )
                )
            case tac.Subscript(var=var):
                targets = tp.pointers[LOCALS, var]
                tp.pointers[targets, tac.Var("*")] = pointed
                tp.dirty.update(
                    make_dirty_from_keys(
                        targets,
                        pythia.dom_concrete.Set[tac.Var].singleton(tac.Var("*")),
                    )
                )
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

        # print(f"Transfer {ins} at {location}")
        # print_debug(ins, tp)
        # print(f"Prev: {tp}")

        # FIX: this removes pointers and make it "bottom" instead of "top"
        for var in tac.gens(ins):
            if var in tp.pointers[LOCALS]:
                del tp.pointers[LOCALS][var]

        match ins:
            case tac.Assign(lhs, expr, or_null):
                (pointed, types, dirty) = self.expr(
                    prev_tp, expr, LocationObject(location), tp
                )
                if or_null and len(pointed.as_set()) == 0:
                    t = ts.literal(ts.NULL)
                    (pointed, types, dirty) = (immutable(t), t, make_dirty())
                tp.dirty.update(dirty)
                self.signature(tp, lhs, pointed, types)
            case tac.Return(var):
                val = tp.pointers[LOCALS, var]
                tp.pointers[LOCALS, tac.Var("return")] = val
        #
        # print_debug(ins, tp)
        # print(f"Post: {tp}")
        # print()

        # tp.normalize_types()
        tp.collect_garbage(self.liveness[location])

        # assert old_prev_tp == prev_tp, f'{old_prev_tp}\n{prev_tp}'
        return tp


def print_debug(ins: tac.Tac, tp: TypedPointer) -> None:
    for var in tac.free_vars(ins):
        if var in tp.pointers[LOCALS].keys():
            p = tp.pointers[LOCALS, var]
            t = tp.types[p]
            print(f"  {var} = {p} : {t}")
        else:
            print(f"  {var} = <bottom>")


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
    for var in tp.dirty[LOCALS].as_set():
        if var.is_stackvar:
            continue
        yield var.name
    alive = liveness.as_set()
    for var, target in tp.pointers[LOCALS].items():
        if var.name == "return":
            continue
        reachable = set(find_reachable(tp.pointers, alive, set(), target.as_set()))
        if var in alive and any(tp.dirty[obj] for obj in reachable):
            if var.is_stackvar:
                continue
            yield var.name

    # TODO: assert that all dirty objects are reachable from locals
    # TODO: assert that only the iterator is reachable from stack variables
