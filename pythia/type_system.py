from __future__ import annotations as _

import ast
import contextlib
import os
import pathlib
import typing
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from pythia.utils import discard as _discard


@dataclass(frozen=True, slots=True)
class Ref:
    name: str

    def __repr__(self) -> str:
        return self.name


ANY = Ref("typing.Any")
LITERAL = Ref("typing.Literal")
LIST = Ref("builtins.list")
TUPLE = Ref("builtins.tuple")
SET = Ref("builtins.set")
TYPE = Ref("builtins.type")
NONE_TYPE = Ref("builtins.NoneType")
BOOL = Ref("builtins.bool")
INT = Ref("builtins.int")
FLOAT = Ref("builtins.float")
STR = Ref("builtins.str")


@dataclass(frozen=True, slots=True)
class TypeVar:
    name: str
    is_args: bool = False

    def __repr__(self) -> str:
        return f"*{self.name}" if self.is_args else self.name


@dataclass(frozen=True, slots=True)
class Star:
    items: tuple[TypeExpr, ...]

    def __repr__(self) -> str:
        return f"*{self.items}"


@dataclass(frozen=True, slots=True)
class Index:
    number: typing.Optional[int]
    name: typing.Optional[str]

    def __repr__(self) -> str:
        if self.number is None:
            if self.name is None:
                return "None"
            return f"{self.name}"
        if self.name is None:
            return f"{self.number}"
        return f"({self.number}){self.name}"

    def __sub__(self, other: int | Index) -> Index:
        if self.number is None:
            return self
        if isinstance(other, Index):
            if other.number is None:
                return self
            other = other.number
        return Index(self.number - other, self.name)

    def __le__(self, other: Index) -> bool:
        if self.number is None or other.number is None:
            return True
        return self.number <= other.number

    def __lt__(self, other: Index) -> bool:
        if self.number is None or other.number is None:
            return False
        return self.number < other.number


@dataclass(frozen=True, slots=True)
class Union:
    items: frozenset[TypeExpr]

    def __repr__(self) -> str:
        if not self.items:
            return "BOT"
        return f'{{{" | ".join(f"{item}" for item in self.items)}}}'


@dataclass(frozen=True, slots=True)
class Row:
    # an index has either a name or an index or both
    index: Index
    type: TypeExpr

    def __repr__(self) -> str:
        return f"{self.index}: {self.type}"

    def __le__(self, other: Row) -> bool:
        return self.index <= other.index

    def __lt__(self, other: Row) -> bool:
        return self.index < other.index


def make_row(
    number: typing.Optional[int], name: typing.Optional[str], t: TypeExpr
) -> Row:
    return Row(
        Index(
            number if number is not None else None, name if name is not None else None
        ),
        t,
    )


NULL = ...


@dataclass(frozen=True, slots=True)
class Literal:
    value: int | str | bool | float | tuple | TypeVar | None | Ellipsis
    ref: Ref

    def __repr__(self) -> str:
        if isinstance(self.value, tuple):
            return f"Literal({list(self.value)})"
        return f"Literal({self.value!r})"


def literal(value: int | str | bool | float | tuple | list | None) -> Literal:
    match value:
        case value if value is NULL:
            ref = Ref("builtins.ellipsis")
        case int():
            ref = INT
        case float():
            ref = FLOAT
        case str():
            ref = STR
        case bool():
            ref = BOOL
        case None:
            ref = NONE_TYPE
        case tuple():
            ref = TUPLE
        case list():
            value = tuple(value)
            ref = LIST
        case _:
            assert False, f"Unknown literal type {value!r}"
    return Literal(value, ref)


NONE = literal(None)


@dataclass(frozen=True, slots=True)
class TypedDict:
    items: frozenset[Row]

    def __repr__(self) -> str:
        items = self.items
        # if not items:
        #     return 'TOP'
        if all(isinstance(item, Row) for item in items):
            res = ", ".join(
                f"{item.index}={item.type}"
                for item in sorted(self.row_items(), key=lambda item: item.index)
            )
            if len(items) == 1:
                return f"({res},)"
            return f"({res})"
        return f'{{{" & ".join(f"{decl}" for decl in items)}}}'

    def row_items(self) -> frozenset[Row]:
        assert all(isinstance(item, Row) for item in self.items)
        return typing.cast(frozenset[Row], self.items)

    def split_by_index(
        self: TypedDict, value: int | str
    ) -> tuple[TypedDict, TypedDict]:
        index = index_from_literal(value)
        items = self.row_items()
        good = typed_dict([item for item in items if match_index(item.index, index)])
        bad = typed_dict([item for item in items if not match_index(item.index, index)])
        return good, bad


@dataclass(frozen=True, slots=True)
class Overloaded:
    items: tuple[FunctionType, ...]

    def __repr__(self) -> str:
        return f"Overloaded({self.items})"

    def all_new(self) -> bool:
        return all(item.new for item in self.items)

    def any_new(self) -> bool:
        return any(item.new for item in self.items)


@dataclass(frozen=True, slots=True)
class Class:
    name: str
    class_dict: TypedDict
    inherits: tuple[TypeExpr, ...]
    protocol: bool
    type_params: tuple[TypeVar, ...]

    def __repr__(self) -> str:
        return f"instance {self.name}"


@dataclass(frozen=True, slots=True)
class Module:
    name: str
    class_dict: TypedDict

    def __repr__(self) -> str:
        return f"module {self.name}"


@dataclass(frozen=True, slots=True)
class SideEffect:
    new: bool
    bound_method: bool = False
    update: tuple[typing.Optional[TypeExpr], tuple[int, ...]] = (None, ())
    points_to_args: bool = False
    alias: tuple[str, ...] = ()  # Field names to copy from self (e.g., ("_buffer",))
    accessor: bool = False  # Returns objects from self's * field (e.g., __getitem__)


@dataclass(frozen=True, slots=True)
class FunctionType:
    params: TypedDict
    return_type: TypeExpr
    side_effect: SideEffect
    is_property: bool
    type_params: tuple[TypeVar, ...]

    def __repr__(self) -> str:
        if self.is_property:
            return f"->{self.return_type}"
        else:
            type_params = ", ".join(str(x) for x in self.type_params)
            new = "new " if self.new() else ""
            update = (
                "{update " + str(self.side_effect.update) + "}@"
                if self.side_effect.update[0]
                else ""
            )
            return f"[{type_params}]({self.params} -> {update}{new}{self.return_type})"

    def new(self):
        return self.side_effect.new


@dataclass(frozen=True, slots=True)
class Instantiation:
    generic: Class | FunctionType | Ref
    type_args: tuple[TypeExpr, ...]

    def __repr__(self) -> str:
        return f'{self.generic}[{", ".join(repr(x) for x in self.type_args)}]'


@dataclass(frozen=True, slots=True)
class Access:
    items: TypeExpr  # Typevar before resolution
    arg: TypeExpr

    def __repr__(self) -> str:
        return f"{self.items}.[{self.arg}]"


type TypeExpr = typing.Union[
    Ref,
    TypeVar,
    Star,
    Literal,
    TypedDict,
    Overloaded,
    Row,
    Class,
    Module,
    FunctionType,
    Union,
    Instantiation,
    Access,
]


def squeeze(t: TypeExpr) -> TypeExpr:
    """Squeeze a type expression to its simplest form."""
    if isinstance(t, FunctionType):
        return replace(
            t,
            return_type=squeeze(t.return_type),
        )
    if isinstance(t, Overloaded):
        items = frozenset(squeeze(item) for item in t.items)
        return Overloaded(tuple(items))
    if isinstance(t, Union):
        items = frozenset(squeeze(item) for item in t.items)
        if len(items) == 1:
            return next(iter(t.items))
        return Union(items)
    if isinstance(t, TypedDict):
        items = frozenset(squeeze(row) for row in t.items)
        if len(items) == 1:
            return next(iter(items))
        return replace(t, items=items)
    if isinstance(t, Instantiation):
        if t.generic == Ref("typing.Union"):
            return union(t.type_args, should_squeeze=True)
        res = replace(
            t,
            type_args=tuple(squeeze(arg) for arg in t.type_args),
            generic=squeeze(t.generic),
        )
        return res
    if isinstance(t, Access):
        return Access(squeeze(t.items), squeeze(t.arg))
    return t


def unpack_star(type_args: typing.Iterable[TypeExpr]) -> tuple[TypeExpr, ...]:
    unpacked_args: list[TypeExpr] = []
    for arg in type_args:
        if isinstance(arg, Star):
            unpacked_args.extend(arg.items)
        else:
            unpacked_args.append(arg)
    return tuple(unpacked_args)


def bind_typevars(t: TypeExpr, context: dict[TypeVar, TypeExpr]) -> TypeExpr:
    if not context:
        return t
    match t:
        case Module() | Ref():
            return t
        case TypeVar() as t:
            return context.get(t, t)
        case Star(items):
            return Star(tuple(bind_typevars(item, context) for item in items))
        case Literal() as t:
            if isinstance(t.value, tuple):
                return replace(
                    t, value=unpack_star(bind_typevars(x, context) for x in t.value)
                )
            return t
        case TypedDict(items):
            return typed_dict(bind_typevars(item, context) for item in items)
        case Overloaded(items):
            return overload(bind_typevars(item, context) for item in items)
        case FunctionType() as f:
            context = {k: v for k, v in context.items() if k not in f.type_params}
            if not context:
                return f
            new_params = bind_typevars(f.params, context)
            new_return_type = bind_typevars(f.return_type, context)
            new_side_effect = bind_typevars(f.side_effect, context)
            return replace(
                f,
                params=new_params,
                return_type=new_return_type,
                side_effect=new_side_effect,
            )
        case Union(items):
            return join_all(bind_typevars(item, context) for item in items)
        case Row() as row:
            return replace(row, type=bind_typevars(row.type, context))
        case SideEffect() as s:
            if s.update[0] is None:
                return s
            return replace(s, update=(bind_typevars(s.update[0], context), s.update[1]))
        case Class(
            type_params=type_params, class_dict=class_dict, inherits=inherits
        ) as klass:
            context = {k: v for k, v in context.items() if k not in type_params}
            if not context:
                return klass
            return replace(
                klass,
                class_dict=bind_typevars(class_dict, context),
                inherits=tuple(bind_typevars(x, context) for x in inherits),
            )
        case Instantiation(generic, type_args) as instantiation:
            new_type_args = unpack_star(bind_typevars(x, context) for x in type_args)
            return replace(
                instantiation,
                generic=bind_typevars(generic, context),
                type_args=new_type_args,
            )
        case Access(argsvar, arg):
            choices = bind_typevars(argsvar, context)
            actual_arg = bind_typevars(arg, context)
            if (
                isinstance(choices, Star)
                and isinstance(actual_arg, Literal)
                and isinstance(actual_arg.value, int)
            ):
                return choices.items[actual_arg.value]
            return Access(choices, actual_arg)
    raise NotImplementedError(f"{t!r}, {type(t)}")


def typed_dict(rows: typing.Iterable[Row]) -> TypedDict:
    rows = tuple(rows)
    assert all(isinstance(x, Row) for x in rows)
    assert len({x.index for x in rows}) == len(rows), rows
    return TypedDict(frozenset(rows))


def overload(functions: typing.Iterable[FunctionType | Overloaded]) -> Overloaded:
    collect: list[FunctionType] = []
    for f in functions:
        if isinstance(f, Overloaded):
            assert all(isinstance(x, FunctionType) for x in f.items)
            collect.extend(f.items)
        elif isinstance(f, FunctionType):
            collect.append(f)
        else:
            assert f == BOTTOM, f"{f!r}"
            raise TypeError("Trying add a bottom function")
    # assert len(collect) > 0
    return Overloaded(tuple(collect))


def union(items: typing.Iterable[TypeExpr], should_squeeze=True) -> TypeExpr:
    items = frozenset(items)
    if any(isinstance(x, Literal) for x in items):
        if any(isinstance(x, Ref) for x in items):
            pass
    # assert len(items) > 0
    res = Union(items)
    if should_squeeze:
        return squeeze(res)
    return res


TOP = typed_dict([])
BOTTOM = Union(frozenset())


def join(t1: TypeExpr, t2: TypeExpr) -> TypeExpr:
    if t1 == t2:
        return t1
    if t1 == BOTTOM:
        return t2
    if t2 == BOTTOM:
        return t1
    if t1 == TOP or t2 == TOP:
        return TOP
    if t1 == ANY or t2 == ANY:
        return ANY
    match t1, t2:
        case Union(items1), Union(items2):
            return squeeze(Union(items1 | items2))
        case (Union(items), other) | (other, Union(items)):  # type: ignore
            return squeeze(Union(items | {other}))
        case (Overloaded() as f, Overloaded() as g):
            if len(f.items) == 0:
                return g
            if len(g.items) == 0:
                return f
            if len(f.items) == 1 and len(g.items) == 1:
                res = join(f.items[0], g.items[0])
                if isinstance(res, FunctionType):
                    return overload([res])
            return union([t1, t2])
        case (Overloaded(), _) | (_, Overloaded()):
            return TOP
        case (FunctionType() as f1, FunctionType() as f2):
            # if (f1.is_property, f1.type_params) == (f2.is_property, f2.type_params):
            #     # TODO: check that f1.params and f2.params are compatible
            #     new_params = meet(f1.params, f2.params)
            #     if isinstance(new_params, Row):
            #         new_params = typed_dict([new_params])
            #     assert isinstance(new_params, TypedDict)
            #     return replace(
            #         f1,
            #         params=new_params,
            #         return_type=join(f1.return_type, f2.return_type),
            #         side_effect=join(f1.side_effect, f2.side_effect),
            #     )
            return union([f1, f2])
        case (Literal() as l1, Literal() as l2):
            if l1.ref == l2.ref:
                if isinstance(l1.value, tuple):
                    if len(l1.value) == len(l2.value):
                        return Literal(
                            tuple(join(t1, t2) for t1, t2 in zip(l1.value, l2.value)),
                            l1.ref,
                        )
                    if l1.ref == LIST:
                        return Instantiation(
                            l1.ref, (join_all([*l1.value, *l2.value]),)
                        )
                return l1.ref
            assert l1 != l2
            return union([t1, t2])
        case (Literal() as n, Ref() as ref) | (
            Ref() as ref,
            Literal() as n,
        ) if n.ref == ref:
            return ref
        case (Literal(tuple() as value, ref=ref), Instantiation() as inst) | (
            Instantiation() as inst,
            Literal(tuple() as value, ref=ref),
        ) if ref == inst.generic:
            if ref == LIST:
                value = (join_all(value),)
            return join(inst, Instantiation(ref, value))
        case (TypedDict(items1), TypedDict(items2)):  # type: ignore
            return TypedDict(items1.intersection(items2))
        case (Ref() as ref, other) | (other, Ref() as ref):  # type: ignore
            if instantiate_static_ref(ref) == other:
                return ref
            # Bare generic Ref subsumes Instantiation of the same class
            if isinstance(other, Instantiation):
                generic = other.generic
                if isinstance(generic, Ref):
                    generic = instantiate_static_ref(generic)
                ref_class = instantiate_static_ref(ref)
                if isinstance(ref_class, Class) and isinstance(generic, Class):
                    if ref_class.name == generic.name and ref_class.type_params:
                        return ref
            return union([ref, other])
        case (
            Instantiation(generic1, type_args1),
            Instantiation(generic2, type_args2),
        ) if generic1 == generic2:
            return Instantiation(
                generic1, tuple(join(t1, t2) for t1, t2 in zip(type_args1, type_args2))
            )
        case (Row(index1, t1), Row(index2, t2)):
            if index1 == index2:
                return Row(index1, join(t1, t2))
            return BOTTOM
        case (Row(_, t1), (Instantiation(Ref("builtins.list"), type_args))) | (
            Instantiation(Ref("builtins.list"), type_args),
            Row(_, t1),
        ):
            return Instantiation(LIST, tuple(join(t1, t) for t in type_args))
        case (Row(_, t1), (Instantiation(Ref("builtins.tuple"), type_args))) | (
            Instantiation(Ref("builtins.tuple"), type_args),
            Row(_, t1),
        ):
            # not exact; should only join at the index of the row
            return Instantiation(TUPLE, tuple(join(t1, t) for t in type_args))
        case Class(), Class():
            return TOP
        case (Class(name="int") | Ref("builtins.int") as c, Literal(int())) | (
            Literal(int()),
            Class(name="int") | Ref("builtins.int") as c,
        ):
            return c
        case (SideEffect() as s1, SideEffect() as s2):
            assert s1.update[1] == s2.update[1]
            return SideEffect(
                new=s1.new | s2.new,
                bound_method=s1.bound_method | s2.bound_method,
                update=(join(s1.update[0], s2.update[0]), s1.update[1]),
                points_to_args=s1.points_to_args | s2.points_to_args,
                alias=s1.alias + tuple(f for f in s2.alias if f not in s1.alias),
            )
        case (Access(items1, arg1), Access(items2, arg2)):
            return Access(join(items1, items2), join(arg1, arg2))
        case x, y:
            return union([x, y])
    raise NotImplementedError(f"{t1!r}, {t2!r}")


def join_all(items: typing.Iterable[TypeExpr]) -> TypeExpr:
    res: TypeExpr = BOTTOM
    for t in unpack_star(items):
        res = join(res, t)
    return res


def meet(t1: TypeExpr, t2: TypeExpr) -> TypeExpr:
    if t1 == t2:
        return t1
    if t1 == TOP:
        return t2
    if t2 == TOP:
        return t1
    if t1 == BOTTOM or t2 == BOTTOM:
        return BOTTOM
    match t1, t2:
        case (Literal() as l1, Literal() as l2):
            if l1.ref == l2.ref:
                if isinstance(l1.value, tuple) and isinstance(l2.value, tuple):
                    if len(l1.value) == len(l2.value):
                        return Literal(
                            tuple(meet(t1, t2) for t1, t2 in zip(l1.value, l2.value)),
                            l1.ref,
                        )
                    if l1.ref == LIST:
                        return Instantiation(
                            l1.ref, (meet_all([*l1.value, *l2.value]),)
                        )
                return l1.ref
            assert l1 != l2
            assert False, f"{l1!r}, {l2!r}"
        case (Ref() as ref, Instantiation() as inst) | (
            Instantiation() as inst,
            Ref() as ref,
        ) if ref == inst.generic:
            return inst
        case (Literal(tuple() as value, ref=ref), Instantiation() as inst) | (
            Instantiation() as inst,
            Literal(tuple() as value, ref=ref),
        ) if ref == inst.generic:
            if ref.name == "builtins.list":
                value = (meet_all(value),)
            return meet(inst, Instantiation(ref, value))
        case (Overloaded(), _):
            return TOP
        case (TypedDict(items1), TypedDict(items2)):  # type: ignore
            return typed_dict(items1 | items2)
        case (Ref(), Ref()):
            return BOTTOM
        case (Ref() as ref, Literal() as n) | (
            Literal() as n,
            Ref() as ref,
        ) if n.ref == ref:
            return n
        case (Union(items), t) | (t, Union(items)):  # type: ignore
            if t in items:
                return t
            types = [meet(item, t) for item in items]
            return join_all(types)
        case (Row(index1, t1), Row(index2, t2)):
            if index1 == index2:
                return Row(index1, meet(t1, t2))
            return TOP
        case (Overloaded(items), FunctionType() as f):
            return overload([*items, f])
        case (FunctionType() as f, Overloaded(items)):
            return overload([f, *items])
        case (FunctionType() as f1, FunctionType() as f2):
            # if (f1.is_property, f1.side_effect, f1.type_params, f1.return_type)
            #       == (f2.is_property, f2.side_effect, f2.type_params, f1.return_type):
            #     new_params = join(f1.params, f2.params)
            #     if not isinstance(new_params, Union):
            #         if isinstance(new_params, Row):
            #             new_params = typed_dict([new_params])
            #         # TODO: side_effect = join(f1.side_effect, f2.side_effect)
            #         assert isinstance(new_params, TypedDict)
            #         return replace(f1, params=new_params,
            #                        return_type=meet(f1.return_type, f2.return_type))
            return overload([f1, f2])
        case (SideEffect() as s1, SideEffect() as s2):
            assert s1.update[1] == s2.update[1]
            return SideEffect(
                new=s1.new & s2.new,
                bound_method=s1.bound_method & s2.bound_method,
                update=(meet(s1.update[0], s2.update[0]), s1.update[1]),
                points_to_args=s1.points_to_args & s2.points_to_args,
                alias=tuple(f for f in s1.alias if f in s2.alias),
            )
        case (Access(items1, arg1), Access(items2, arg2)):
            return Access(meet(items1, items2), meet(arg1, arg2))
        case (t1, t2):
            raise NotImplementedError(f"{t1!r}, {t2!r}")
    raise AssertionError


def meet_all(items: typing.Iterable[TypeExpr]) -> TypeExpr:
    res: TypeExpr = TOP
    for t in unpack_star(items):
        res = meet(res, t)
    return res


def is_subtype(left: TypeExpr, right: TypeExpr) -> bool:
    if right == ANY:
        return True
    if left == ANY:
        return True
    return join(left, right) == right


def index_from_literal(index: int | str) -> Index:
    return Index(index, None) if isinstance(index, int) else Index(None, index)


def match_index(param: Index, arg: Index) -> bool:
    return (
        arg.number == param.number
        and arg.number is not None
        or arg.name == param.name
        and arg.name is not None
    )


def unify_argument(
    type_params: tuple[TypeVar, ...], param: TypeExpr, arg: TypeExpr
) -> typing.Optional[dict[TypeVar, TypeExpr]]:
    if param == arg:
        return {}
    if arg == BOTTOM:
        return {x: BOTTOM for x in (set(type_params) & free_vars_expr(param))}
    if param == TOP:
        return {}
    match param, arg:
        case TypeVar() as param, arg:
            return {param: arg}
        case (Ref("typing.Any"), _) | (_, Ref("typing.Any")):
            return {}
        case Union(items), arg:
            res = [
                unified
                for item in items
                if (unified := unify_argument(type_params, item, arg)) is not None
            ]
            if not res:
                return None
            present_keys = {k for t in res for k in t}
            return {k: join_all(t.get(k, BOTTOM) for t in res) for k in type_params if k in present_keys}
        case param, TypedDict(items):
            res = [
                unified
                for item in items
                if (unified := unify_argument(type_params, param, item)) is not None
            ]
            if not res:
                return None
            present_keys = {k for t in res for k in t}
            return {k: join_all(t.get(k, BOTTOM) for t in res) for k in type_params if k in present_keys}
        case Class() as param, Instantiation(Class() as arg, arg_args):
            if param.name == arg.name:
                return {k: v for k, v in zip(param.type_params, arg_args)}
            return None
        case Instantiation() as param, Class() as arg if arg.type_params:
            generic = param.generic
            if isinstance(generic, Ref):
                generic = instantiate_static_ref(generic)
            if not isinstance(generic, Class) or generic.name != arg.name:
                return None
            # Bare generic class: unify type args against ANY (unknown type params)
            ps = typed_dict([make_row(i, None, p) for i, p in enumerate(param.type_args)])
            args_td = typed_dict([make_row(i, None, ANY) for i in range(len(arg.type_params))])
            mid_context = unify(type_params, ps, args_td)
            if mid_context is None:
                return None
            return mid_context.bound_typevars
        case Instantiation() as param, Instantiation() as arg:
            if param.generic != arg.generic:
                param_class = instantiate_static_ref(param.generic)
                if not isinstance(param_class, Class) or not param_class.protocol:
                    return None
                param_dict = param_class.class_dict

                arg_class = instantiate_static_ref(arg.generic)
                if not isinstance(arg_class, Class):
                    return None
                arg_dict = arg_class.class_dict

                unified = []
                for param_row in param_dict.row_items():
                    for arg_row in arg_dict.row_items():
                        if param_row.index.name == arg_row.index.name:
                            param_f = bind_self_function(
                                overload([param_row.type]), param
                            )
                            assert isinstance(param_f, Overloaded)
                            param_f = param_f.items[0]
                            arg_f = bind_self_function(overload([arg_row.type]), arg)
                            assert isinstance(arg_f, Overloaded)
                            arg_f = arg_f.items[0]
                            res = unify_argument(type_params, param_f, arg_f)
                            if res is None:
                                return None
                            unified.append(res)
                            break

                if len(unified) != 1:
                    return None
                return unified[0]
            ps = typed_dict(
                [make_row(i, None, p) for i, p in enumerate(param.type_args)]
            )
            args = typed_dict(
                [make_row(i, None, p) for i, p in enumerate(arg.type_args)]
            )
            mid_context = unify(type_params, ps, args)
            if mid_context is None:
                return None
            return mid_context.bound_typevars
        case Instantiation() as param, Literal(tuple() as value, ref):
            if param.generic != ref:
                return None
            if ref.name == "builtins.list":
                value = (join_all(value),)
            return unify_argument(type_params, param, Instantiation(ref, value))
        case param, Instantiation(Ref() as param_type, param_args) if not isinstance(
            param, Ref
        ):
            return unify_argument(
                type_params,
                param,
                Instantiation(instantiate_static_ref(param_type), param_args),
            )
        case FunctionType() as param, FunctionType() as arg:
            if param.is_property != arg.is_property:
                return None
            if param.side_effect != arg.side_effect:
                return None
            mid_context = unify_argument(type_params, param.params, arg.params)
            if mid_context is None:
                return None
            res = unify_argument(type_params, param.return_type, arg.return_type)
            return res
        case Literal() as param, Literal() as arg:
            if param == arg:
                return {}
            if param.ref == arg.ref and isinstance(param.value, tuple):
                assert isinstance(arg.value, tuple)
                mid_context = unify(
                    type_params,
                    typed_dict(
                        [make_row(i, None, p) for i, p in enumerate(param.value)]
                    ),
                    typed_dict([make_row(i, None, p) for i, p in enumerate(arg.value)]),
                )
                assert mid_context is not None
                return mid_context.bound_typevars
            return None
        case Row() as param, Row() as arg:
            if match_index(param.index, arg.index):
                return unify_argument(type_params, param.type, arg.type)
            return None
        case (Class("int") | Ref("builtins.int"), Literal(int())) | (
            Literal(int()),
            Class("int") | Ref("builtins.int"),
        ):
            return {}
        case Class(), Literal():
            return None
        case param, Ref() as arg:
            return unify_argument(type_params, param, instantiate_static_ref(arg))
        case Ref() as param, Literal(ref=arg) if arg == param:
            return {}
        case Ref() as param, arg:
            return unify_argument(type_params, instantiate_static_ref(param), arg)
        case param, arg:
            return None
    assert False


@dataclass
class Binding:
    bound_typevars: dict[TypeVar, TypeExpr]
    bound_params: dict[Row, Row]
    unbound_params: set[Row]


def unify(
    type_params: tuple[TypeVar, ...], params: TypedDict, args: TypedDict
) -> typing.Optional[Binding]:
    bound_types: dict[TypeVar, TypeExpr] = {}
    bound_params: dict[Row, Row] = {}
    unbound_params: set[Row] = set(params.row_items())
    unbound_args = set(args.row_items())
    varparam: typing.Optional[Row] = None
    for param in sorted(params.row_items()):
        if isinstance(param.type, TypeVar) and param.type.is_args:
            varparam = param
            continue
        matching_args = {
            arg for arg in unbound_args if match_index(param.index, arg.index)
        }
        if not matching_args:
            continue
        unbound_params.remove(param)
        unbound_args -= matching_args
        assert len(matching_args) == 1, f"{matching_args!r}"
        matching_arg = matching_args.pop()
        binding = unify_argument(type_params, param.type, matching_arg.type)
        if binding is None:
            return None
        bound_params[param] = matching_arg
        bindable_type_params = {k for k in type_params if k in binding}
        for k in bindable_type_params:
            if k.is_args:
                items: set[TypeExpr] = {
                    bound_types.get(k, None),
                    binding.get(k, None),
                } - {None}
                assert len(items) <= 1, f"{items!r}"
                if items:
                    bound_types[k] = items.pop()
            else:
                bound_types[k] = join(
                    bound_types.get(k, BOTTOM), binding.get(k, BOTTOM)
                )
    if varparam is not None:
        # varparam is never removed from unbound_params, since it can continue matching
        elements: list[TypeExpr] = []
        if unbound_args:
            by_index: dict[int, list[TypeExpr]] = {}
            for arg in unbound_args:
                assert arg.index.number is not None
                by_index.setdefault(arg.index.number, []).append(arg.type)
            unbound_args.clear()
            minimal_index = min(by_index)
            maximal_index = max(by_index)
            for i in range(minimal_index, maximal_index + 1):
                if i not in by_index:
                    break
                elements.append(join_all(by_index[i]))
                del by_index[i]
            if by_index:
                return None
        assert isinstance(varparam.type, TypeVar)
        elements = [bind_typevars(e, bound_types) for e in elements]
        bound_types[varparam.type] = Star(tuple(elements))
        bound_params[varparam] = Row(index=varparam.index, type=Star(tuple(elements)))
    if unbound_args:
        return None
    return Binding(
        bound_typevars=bound_types,
        bound_params={
            k: bind_typevars(v, bound_types) for k, v in bound_params.items()
        },
        unbound_params=unbound_params,
    )


def match_row(row: Row, arg: Literal | Ref) -> bool:
    match arg:
        case Literal(value):
            return match_index(row.index, index_from_literal(value))
        case Ref(name):
            return name == "builtins.int" and row.index.number is not None
    assert False, f"{row!r}, {arg!r}"


def access(t: TypeExpr, arg: TypeExpr) -> Overloaded | Module:
    if t == ANY:
        return TOP
    match t, arg:
        case Class(class_dict=class_dict) | Module(class_dict=class_dict), Literal(
            str() as value
        ):
            good, bad = class_dict.split_by_index(value)
            types: list[TypeExpr] = [x.type for x in good.row_items()]
            if len(types) == 1:
                [t] = types
                if isinstance(t, Module):
                    return t
                if not isinstance(t, (Overloaded, FunctionType)):
                    return t
            if (
                not types
                and isinstance(t, Module)
                and t.name not in ["typing", "builtins", "numpy", "collections"]
            ):
                return BOTTOM
            result = overload(types)
            assert (
                free_vars_expr(result) == set()
            ), f"{result!r}, {free_vars_expr(result)}"
            return result
        case Class() as t, arg:
            getter = access(t, literal("__getitem__"))
            getitem = partial(
                getter, typed_dict([make_row(1, None, arg)]), only_callable_empty=False
            )
            if getitem == BOTTOM:
                return BOTTOM
            assert isinstance(getitem, Overloaded), f"{getitem!r}"
            if not getitem.items:
                return BOTTOM
            # assert free_vars_expr(getitem) == set(), f'{getitem!r} has free vars: {free_vars_expr(getitem)}'
            if isinstance(getitem, Union):
                assert len(getitem.items) == 1, f"{getitem!r}"
                [getitem] = getitem.items
            assert isinstance(getitem, Overloaded), f"{getitem!r}"
            return overload([replace(g, is_property=True) for g in getitem.items])
        case Module() as t, arg:
            return access(t.class_dict, arg)
        case Ref() as ref, arg:
            return access(instantiate_static_ref(ref), arg)
        case t, TypedDict(items):
            assert False, f"{t!r}, {arg!r}"
        case t, Overloaded(items):
            assert False, f"{t!r}, {arg!r}"
        case Overloaded(tuple()), arg:
            return TOP
        case Union(items), arg:
            return join_all(access(item, arg) for item in items)
        case TypedDict(items), arg:
            # intersect and not meet to differentiate between
            # a non-existent attribute and an attribute with type TOP
            return overload([row.type for row in items if match_row(row, arg)])
        case Literal(ref=ref), arg:
            return access(ref, arg)
        case t, Union(items):
            return overload([access(t, item) for item in items])
        case Instantiation() as t, arg:
            # ignore type arguments for now
            return access(t.generic, arg)
        case _:
            raise NotImplementedError(f"{t=}, {arg=}, {type(t)=}, {type(arg)=}")


def subtract_indices(unbound_params: TypedDict, bound_params: TypedDict) -> TypedDict:
    # Fix: subtract index from all indexable params
    # For this, we need to go from lowest to highest,
    # and subtract from it the number of lower-indexed params that became bound
    # (i.e. the number of params that were removed from the row)
    # Here we assume that the params are consecutive, starting from 0
    indexes = [
        item.index for item in bound_params.row_items() if item.index.number is not None
    ]

    if indexes:
        max_bound = max(indexes)
        unbound_params = typed_dict(
            [
                (
                    replace(item, index=item.index - max_bound - 1)
                    if item.index > max_bound
                    else item
                )
                for item in unbound_params.row_items()
            ]
        )
    return unbound_params


def union_all(iterable: typing.Iterable[set]) -> set:
    result = set()
    for item in iterable:
        result |= item
    return result


def free_vars_expr(t: TypeExpr) -> set[TypeVar]:
    match t:
        case None:
            return set()
        case TypeVar() as t:
            return {t}
        case Star() as t:
            return union_all(free_vars_expr(item) for item in t.items)
        case Access(t, arg):
            return free_vars_expr(t) | free_vars_expr(arg)
        case Ref() | Literal():
            return set()
        case Union(items):
            return union_all(free_vars_expr(item) for item in items)
        case Instantiation(generic, args):
            return free_vars_expr(generic) | union_all(
                free_vars_expr(arg) for arg in args
            )
        case TypedDict() as t:
            return free_vars_typed_dict(t)
        case FunctionType() as f:
            result = free_vars_expr(f.return_type) | free_vars_expr(f.params)
            result -= set(f.type_params) & free_vars_expr(f.params)
            return result
        case Class():
            return set()
        case Module():
            return set()
        case Overloaded() as t:
            return union_all(free_vars_expr(item) for item in t.items)
        case _:
            assert False, f"{t!r} {type(t)!r}"


def free_vars_typed_dict(t: TypedDict) -> set[TypeVar]:
    return union_all(free_vars_expr(item.type) for item in t.row_items())


def subtract_type_underapprox(argtype: TypeExpr, paramtype: TypeExpr) -> TypeExpr:
    if argtype == paramtype:
        return BOTTOM
    if is_subtype(argtype, paramtype):
        return BOTTOM
    if isinstance(argtype, Union) and paramtype in argtype.items:
        return union(argtype.items - {paramtype})
    if isinstance(argtype, Union) and isinstance(paramtype, Union):
        return union(argtype.items - paramtype.items)
    return argtype


def is_type_type(t: TypeExpr) -> bool:
    match t:
        case Instantiation(Ref("builtins.type"), (_,)):
            return True
        case _:
            return False


def get_init_func(callable: TypeExpr) -> TypeExpr:
    if callable == TOP:
        return TOP
    if callable == BOTTOM:
        return BOTTOM
    match callable:
        case Instantiation(Ref("builtins.type"), (selftype,)):
            assert isinstance(selftype, (Ref, Instantiation))
            init = access(selftype, literal("__init__"))
            if init == overload([]):
                return_type = selftype
                if isinstance(selftype, Ref):
                    return_type = Instantiation(selftype, (BOTTOM,))
                else:
                    assert isinstance(selftype.generic, Ref)
                return overload(
                    [
                        FunctionType(
                            params=typed_dict([]),
                            return_type=return_type,
                            side_effect=SideEffect(new=True),
                            is_property=False,
                            type_params=(),
                        )
                    ]
                )
            side_effect = SideEffect(
                new=True, bound_method=True, update=(None, ()), points_to_args=True
            )
            res = bind_self(
                overload(
                    replace(
                        f,
                        return_type=f.side_effect.update[0] or selftype,
                        side_effect=side_effect,
                    )
                    for f in init.items
                ),
                selftype,
            )
            return res
    assert False, f"{callable!r}"


def get_declared_fields(t: TypeExpr) -> list[tuple[str, TypeExpr]]:
    """Return declared instance fields (non-method attributes with type annotations).

    For types with field declarations like `_buffer: object`, returns [('_buffer', object_type)].
    Used to create transitive objects when @new creates instances.
    """
    match t:
        case Class(class_dict=class_dict):
            result = []
            for row in class_dict.row_items():
                # Fields have string names and non-method types
                if row.index.name is not None:
                    # Skip methods (FunctionType, Overloaded)
                    if not isinstance(row.type, (FunctionType, Overloaded)):
                        result.append((row.index.name, row.type))
            return result
        case Ref(name):
            resolved = resolve_static_ref(t)
            if resolved != t:
                return get_declared_fields(resolved)
            return []
        case Instantiation(generic, type_args):
            # For type[Class], get fields from the Class
            if isinstance(generic, Ref) and generic.name == "builtins.type":
                if type_args and isinstance(type_args[0], Class):
                    return get_declared_fields(type_args[0])
            return get_declared_fields(generic)
        case _:
            return []


def do_bind(f: FunctionType, binding: Binding) -> FunctionType:
    # FIX: partially-bind *Args, so when *Args is bound to (T1, T2),
    # then Literal[*Args] becomes Literal[T1, T2, *Args]
    side_effect = bind_typevars(f.side_effect, binding.bound_typevars)
    assert isinstance(side_effect, SideEffect)
    params = subtract_indices(
        typed_dict(binding.unbound_params), typed_dict(binding.bound_params.keys())
    )
    stars = {e for e in free_vars_expr(params) if isinstance(e, TypeVar) and e.is_args}
    return_type = bind_typevars(f.return_type, binding.bound_typevars)
    type_params = tuple(
        v for v in f.type_params if v not in binding.bound_typevars or v in stars
    )
    res = FunctionType(
        type_params=type_params,
        params=params,
        return_type=return_type,
        side_effect=side_effect,
        is_property=f.is_property,
    )
    # bind params only when type parameters shadow the *Args
    res = bind_typevars(res, binding.bound_typevars)
    assert isinstance(res, FunctionType)
    return res


def split_by_args(callable: TypeExpr, args: TypedDict) -> TypeExpr:
    if callable == TOP:
        return TOP
    if callable == BOTTOM:
        return BOTTOM
    match callable:
        case Overloaded(items):
            bind_list = defaultdict(list)
            for j, f in enumerate(items):
                b = unify(f.type_params, f.params, args)
                if b is not None:
                    key = typed_dict(b.bound_params.keys())
                    bind_list[key].append(do_bind(f, b))
            side_effect = SideEffect(new=True)
            res = overload(
                [
                    FunctionType(
                        params=bound,
                        return_type=squeeze(overload(fs)),
                        is_property=False,
                        type_params=(),
                        side_effect=side_effect,
                    )
                    for bound, fs in bind_list.items()
                ]
            )
            return res
        case FunctionType() as f:
            return split_by_args(overload([f]), args)
        case Union(items):
            return union([split_by_args(item, args) for item in items])
        case Class(class_dict=class_dict), arg:
            dunder = subscr(class_dict, literal("__call__"))
            return split_by_args(dunder, arg)
        case _:
            assert False, f"Cannot call {callable} with {args}"


def partial(callable: TypeExpr, args: TypedDict, only_callable_empty: bool) -> TypeExpr:
    split = split_by_args(callable, args)
    if not isinstance(split, Overloaded):
        return TOP
    if not split.items:
        return BOTTOM
    for item in split.items:
        result = item.return_type
        assert isinstance(result, Overloaded), f"{item!r}"
        # assert not free_vars_expr(result), f"{result!r}"
        if only_callable_empty:
            result = overload([f for f in result.items if is_callable_empty(f)])
            if not result.items:
                continue
        return result
    return BOTTOM


def positional(*args: TypeExpr) -> TypedDict:
    return typed_dict([make_row(i, None, arg) for i, arg in enumerate(args)])


def bind_self_function(f: Overloaded, selftype: TypeExpr) -> Overloaded:
    split_f = split_by_args(f, positional(selftype))
    if not isinstance(split_f, Overloaded):
        return TOP
    res = []
    for item in split_f.items:
        assert isinstance(item.return_type, Overloaded)
        res.append(item.return_type)
    return overload(
        [
            replace(f, side_effect=replace(f.side_effect, bound_method=True))
            for item in res
            for f in item.items
        ]
    )


def bind_self(attr: Overloaded, selftype: TypeExpr) -> Overloaded:
    if isinstance(selftype, Module):
        return attr
    if isinstance(selftype, Ref) and isinstance(resolve_static_ref(selftype), Module):
        return attr
    return bind_self_function(attr, selftype)


def is_callable_empty(f: FunctionType | Overloaded) -> bool:
    match f:
        case Overloaded() as f:
            return all(is_callable_empty(item) for item in f.items)
        case FunctionType(params=TypedDict(items)):
            if not items:
                return True
            if len(items) == 1:
                [row] = items
                return isinstance(row.type, TypeVar) and row.type.is_args
    return False


def get_return(callable: TypeExpr) -> TypeExpr:
    if callable == TOP:
        return TOP
    if callable == BOTTOM:
        return BOTTOM
    match callable:
        case Overloaded(items):
            for f in items:
                if is_callable_empty(f):
                    return f.return_type
            return BOTTOM
        case Union(items):
            res = [get_return(item) for item in items]
            return join_all(res)
        case FunctionType() as f:
            if is_callable_empty(f):
                return f.return_type
            return BOTTOM
    assert False, f"{callable!r}"


def subscr(selftype: TypeExpr, index: TypeExpr) -> TypeExpr:
    if selftype == TOP:
        return TOP
    if selftype == BOTTOM:
        return BOTTOM
    match selftype, index:
        case Instantiation(Ref("builtins.type"), (arg1,)), Instantiation(
            Ref("builtins.type"), (arg2,)
        ):
            inner = Instantiation(
                arg1,
                (arg2,),
            )
            return Instantiation(Ref("builtins.type"), (inner,))
    attr_type = access(selftype, index)
    if attr_type == BOTTOM:
        # non-existent attribute
        return BOTTOM
    try:
        result = bind_self(attr_type, selftype)
    except TypeError as ex:
        ex.add_note(f"for c[{index!r}] where c is {selftype!r}")
        raise
    if isinstance(result, FunctionType):
        result = overload([result])
    assert not free_vars_expr(result), f"{result!r}"
    return result


def subscr_get_property(selftype: TypeExpr, index: TypeExpr) -> TypeExpr:
    res = subscr(selftype, index)
    if isinstance(res, Overloaded) and any(f.is_property for f in res.items):
        assert all(f.is_property for f in res.items)
        return get_return(res)
    return res


def call(callable: TypeExpr, args: TypedDict) -> TypeExpr:
    if is_type_type(callable):
        callable = get_init_func(callable)
    f = partial(callable, args, only_callable_empty=True)
    res = get_return(f)
    res = squeeze(res)
    return res


def make_list_constructor() -> Overloaded:
    args = TypeVar("Args", is_args=True)
    return_type = literal([args])
    return overload(
        [
            FunctionType(
                params=typed_dict([make_row(0, "args", args)]),
                return_type=return_type,
                side_effect=SideEffect(new=True, points_to_args=True),
                is_property=False,
                type_params=(args,),
            )
        ]
    )


def make_set_constructor() -> Overloaded:
    args = TypeVar("Args", is_args=True)
    RES = Instantiation(Ref("typing.Union"), (args,))
    ret = overload(
        [
            FunctionType(
                type_params=(),
                params=typed_dict([]),
                return_type=Instantiation(SET, (union([]),)),
                side_effect=SideEffect(new=True, points_to_args=True),
                is_property=False,
            ),
            FunctionType(
                type_params=(args,),
                params=typed_dict([make_row(0, "items", args)]),
                return_type=Instantiation(SET, (RES,)),
                side_effect=SideEffect(new=True, points_to_args=True),
                is_property=False,
            ),
        ]
    )
    return ret


def make_tuple_constructor() -> Overloaded:
    args = TypeVar("Args", is_args=True)
    return_type = Instantiation(TUPLE, (args,))
    return overload(
        [
            FunctionType(
                params=typed_dict([make_row(0, "args", args)]),
                return_type=return_type,
                side_effect=SideEffect(new=True, points_to_args=True),
                is_property=False,
                type_params=(args,),
            )
        ]
    )


def make_slice_constructor() -> Overloaded:
    return_type = Ref("builtins.slice")
    both = union([NONE, INT])
    return overload(
        [
            FunctionType(
                params=typed_dict(
                    [make_row(0, "start", both), make_row(1, "end", both)]
                ),
                return_type=return_type,
                side_effect=SideEffect(new=True),
                is_property=False,
                type_params=(),
            )
        ]
    )


def make_dict_constructor() -> Overloaded:
    """Constructor for dict literals.

    For BUILD_CONST_KEY_MAP: first arg is keys tuple, rest are values.
    For BUILD_MAP: args are key1, value1, key2, value2, ...
    Returns dict[K, V] where K and V are unions of the key/value types.
    """
    args = TypeVar("Args", is_args=True)
    # Simplified: returns dict[Any, union of value types]
    # A more precise implementation would track key-value associations
    return_type = Instantiation(Ref("builtins.dict"), (ANY, ANY))
    return overload(
        [
            FunctionType(
                type_params=(args,),
                params=typed_dict([make_row(0, "args", args)]),
                return_type=return_type,
                side_effect=SideEffect(new=True, points_to_args=True),
                is_property=False,
            )
        ]
    )


def binop_to_dunder_method(op: str) -> tuple[str, typing.Optional[str]]:
    match op:
        case "in":
            return "__contains__", None
        case "+":
            return "__add__", "__radd__"
        case "-":
            return "__sub__", "__rsub__"
        case "*":
            return "__mul__", "__rmul__"
        case "/":
            return "__truediv__", "__rtruediv__"
        case "//":
            return "__floordiv__", "__rfloordiv__"
        case "%":
            return "__mod__", "__rmod__"
        case "**":
            return "__pow__", "__rpow__"
        case "<<":
            return "__lshift__", "__rlshift__"
        case ">>":
            return "__rshift__", "__rrshift__"
        case "&":
            return "__and__", "__rand__"
        case "|":
            return "__or__", "__ror__"
        case "^":
            return "__xor__", "__rxor__"
        case "@":
            return "__matmul__", "__rmatmul__"
        case "==":
            return "__eq__", None
        case "!=":
            return "__ne__", None
        case "<":
            return "__lt__", None
        case "<=":
            return "__le__", None
        case ">":
            return "__gt__", None
        case ">=":
            return "__ge__", None
        case _:
            raise NotImplementedError(f"{op!r}")


def unop_to_dunder_method(op: str) -> str:
    match op:
        case "bool":
            return "__bool__"
        case "-":
            return "__neg__"
        case "+":
            return "__pos__"
        case "~":
            return "__invert__"
        case "not":
            return "__bool__"
        case "iter":
            return "__iter__"
        case "yield iter":
            return "__iter__"
        case "next":
            return "__next__"
        case _:
            raise NotImplementedError(f"{op!r}")


def get_binop(left: TypeExpr, right: TypeExpr, op: str) -> TypeExpr:
    lop, rop = binop_to_dunder_method(op)
    left_ops = access(left, literal(lop))
    if left_ops == TOP:
        return TOP
    if isinstance(left_ops, FunctionType):
        left_ops = overload([left_ops])
    assert isinstance(left_ops, Overloaded), f"{left_ops!r}"
    if left == right:
        return bind_self(left_ops, left)
    right_ops = access(right, literal(rop))
    if right_ops == TOP:
        return TOP
    if isinstance(right_ops, FunctionType):
        right_ops = overload([right_ops])
    if right_ops != BOTTOM:
        assert isinstance(right_ops, Overloaded), f"{right_ops!r}"
        right_ops = overload(swap_binop_params(rf) for rf in right_ops.items)
        ops = overload([left_ops, right_ops])
    else:
        ops = left_ops
    result = bind_self(ops, left)
    assert not free_vars_expr(result), f"{result!r}"
    return result


def swap_binop_params(rf: FunctionType) -> FunctionType:
    assert len(rf.params.items) == 2
    left, right = sorted(rf.params.items)
    right_ops = replace(
        rf,
        params=typed_dict(
            [
                replace(right, index=index_from_literal(0)),
                replace(left, index=index_from_literal(1)),
            ]
        ),
    )
    return right_ops


def partial_binop(left: TypeExpr, right: TypeExpr, op: str) -> TypeExpr:
    binop_func = get_binop(left, right, op)
    if binop_func == BOTTOM:
        # assume there is an implementation.
        return TOP
    if binop_func == TOP:
        return TOP
    result = split_by_args(binop_func, positional(right))
    assert isinstance(result, Overloaded), f"{result!r}"
    return overload([f.return_type for f in result.items])


def get_unop(left: TypeExpr, op: str) -> TypeExpr:
    return subscr(left, literal(unop_to_dunder_method(op)))


def resolve_static_ref(ref: Ref) -> TypeExpr:
    return resolve_relative_ref(ref, MODULES)


def instantiate_static_ref(ref: Ref) -> TypeExpr:
    t = resolve_static_ref(ref)
    if isinstance(t, (Module, Overloaded)):
        return t
    assert isinstance(t, Instantiation), f"{t!r}"
    assert t.generic == Ref("builtins.type")
    return t.type_args[0]


def resolve_relative_ref(ref: Ref, module: Module) -> TypeExpr:
    result: TypeExpr = module
    for attr in ref.name.split("."):
        result = subscr(result, literal(attr))
    return result


def infer_self(row: Row) -> Row:
    function = row.type
    if not isinstance(function, FunctionType):
        return row
    params = function.params
    if not params:
        raise TypeError(f"Cannot bind self to {function}")
    self_args, other_args = params.split_by_index(0)
    [self_arg_row] = self_args.row_items()
    type_params = function.type_params
    if self_arg_row.type == TOP:
        self_type = TypeVar("Self")
        self_arg_row = replace(self_arg_row, type=self_type)
        type_params = (self_type, *type_params)
    g = replace(
        function,
        params=typed_dict([self_arg_row, *other_args.row_items()]),
        type_params=type_params,
    )
    return Row(row.index, g)


def pretty_print_type(t: Module | TypeExpr, indent: int = 0, max_len: int = 0) -> None:
    match t:
        case TypedDict(items):
            max_len = max([len(str(row.index)) for row in items], default=0)
            for row in items:
                pretty_print_type(row, indent, max_len=max_len)
        case Overloaded(items):
            for row in items:
                pretty_print_type(row, indent)
        case Row(index, Instantiation(Ref("builtins.type"), (typeexpr,))) | Row(
            index, typeexpr
        ):
            if index.name is None:
                print(" " * indent, end="")
            else:
                print(" " * indent, index.name, "=", end="" * max_len, sep="")
            if isinstance(typeexpr, (Overloaded, TypedDict)):
                print()
            elif not isinstance(typeexpr, (Class, Module)):
                print(" " * (max_len - len(str(index.name))), end="")
            pretty_print_type(typeexpr, indent)
        case FunctionType(params, return_type, side_effect, is_property, type_params):
            # pretty_params = ', '.join(f'{row.index.name}: {row.type}'
            #                           for row in sorted(params.row_items(), key=lambda x: x.index))
            pretty_type_params = ", ".join(str(x) for x in type_params)
            params_str = f"({params})" if not is_property else ""
            print(
                f'[{pretty_type_params}]{params_str} -> {"new " if side_effect.new else ""}{return_type}'
            )
        case Class(
            name,
            class_dict=class_dict,
            inherits=inherits,
            protocol=protocol,
            type_params=type_params,
        ):
            kind = " protocol" if protocol else ""
            pretty_type_params = ", ".join(str(x) for x in type_params)
            print(
                f'{kind} {name}[{pretty_type_params}]({", ".join(str(x) for x in inherits)})'
            )
            pretty_print_type(class_dict, indent + 4)
        case Module(name, typeexpr):
            print(f"module {name}:", sep="")
            pretty_print_type(typeexpr, indent + 4)
        case Ref(name):
            print(f"{name}")
        case Index(name, number):
            if number is None:
                print(f"{name}")
            elif name is None:
                print(f"{number}")
            else:
                print(f"({number}){name}")
        case Literal(value):
            print(f"{value}", end="")
        case Instantiation(Ref("builtins.type"), (arg,)):
            print("class", end="")
            pretty_print_type(arg, indent)
        case _:
            raise NotImplementedError(f"{t!r}, {type(t)}")


# def parse_side_effect(stmt: ast.stmt) -> SideEffect:
#     assert isinstance(stmt, ast.Assign)
#     assert len(stmt.targets) == 1
#     target = stmt.targets[0]
#     assert isinstance(target, ast.Name)
#
#     return SideEffect(
#         new="new" in name_decorators and not is_immutable(returns),
#         update=update_type,
#         points_to_args="points_to_args" in name_decorators,
#     )


def make_typevar(t: ast.TypeVar | ast.TypeVarTuple) -> TypeVar:
    return TypeVar(t.name, is_args=isinstance(t, ast.TypeVarTuple))


class SymbolTable:
    def __init__(self, scope_name: str | None, parent: SymbolTable | None = None):
        self.generic_vars: dict[str, TypeVar] = {}
        self.aliases: dict[str, str] = {}
        self.names: set[str] = set()
        self.scope_name: str = scope_name
        self.parent: SymbolTable = parent

    def add_generics(
        self, *vars: ast.TypeVar | ast.TypeVarTuple
    ) -> tuple[TypeVar, ...]:
        typevars = [make_typevar(var) for var in vars]
        for var, t in zip(vars, typevars):
            self.generic_vars[var.name] = t
        return tuple(typevars)

    def lookup(self, name: str) -> TypeExpr:
        if name in self.generic_vars:
            return self.generic_vars[name]
        if name in self.aliases:
            return Ref(self.aliases[name])
        if name in self.names:
            assert self.parent is None
            # FIX: give magic name to avoid aliasing
            return Ref(f"{self.scope_name}.{name}")
        if self.parent is not None:
            return self.parent.lookup(name)
        return Ref(f"builtins.{name}")


class TypeExpressionParser(ast.NodeVisitor):
    def __init__(self, symtable: SymbolTable):
        self.symtable = symtable

    def to_type(self, expr: ast.expr) -> TypeExpr:
        return self.visit(expr)

    def visit_None(self, expr: None) -> TypeExpr:
        return ANY

    def visit_Constant(self, value: ast.Constant) -> TypeExpr:
        return literal(value.value)

    def visit_Name(self, name) -> TypeExpr:
        return self.symtable.lookup(name.id)

    def visit_Starred(self, starred: ast.Starred) -> TypeExpr:
        if isinstance(starred.value, ast.Name):
            return TypeVar(starred.value.id, is_args=True)
        return Star((self.to_type(starred.value),))

    def visit_Subscript(self, subscr: ast.Subscript) -> TypeExpr:
        generic = self.to_type(subscr.value)
        if isinstance(subscr.value, ast.Name):
            if subscr.value.id == "Literal":
                assert isinstance(subscr.slice, ast.Constant)
                return self.visit_Constant(subscr.slice)
        if isinstance(generic, TypeVar):
            return Access(generic, self.to_type(subscr.slice))
        match subscr.slice:
            case ast.Tuple(elts=[arg, ast.Constant(value=x)]) if str(x) == "Ellipsis":
                raise NotImplementedError(f"{generic}[{self.to_type(arg)}, ...]")
            case ast.Tuple(elts=elts):
                items = tuple(self.to_type(x) for x in elts)
            case expr:
                items = (self.to_type(expr),)
        return Instantiation(generic, items)

    def visit_Attribute(self, attribute: ast.Attribute) -> TypeExpr:
        ref: TypeExpr = self.to_type(attribute.value)
        assert isinstance(
            ref, Ref
        ), f"Expected Ref, got {ref!r} for {attribute.value!r}"
        return Ref(f"{ref.name}.{attribute.attr}")

    def visit_BinOp(self, binop: ast.BinOp) -> TypeExpr:
        left_type = self.to_type(binop.left)
        right_type = self.to_type(binop.right)
        if isinstance(binop.op, ast.BitOr):
            return union([left_type, right_type])
        raise NotImplementedError(f"{left_type} {binop.op} {right_type}")


def free_vars(node: ast.expr | ast.arg) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def is_immutable(value: TypeExpr) -> bool:
    match value:
        case Module() as module:
            return all(is_immutable(x.type) for x in module.class_dict.items)
        case Overloaded(items) | Union(items):
            return all(is_immutable(x) for x in items)
        case TypedDict(items):
            if value == TOP:
                return False
            return all(is_immutable(x) for x in items.values())
        case Instantiation(Ref("builtins.type"), (_,)):
            return True  # Not really, but we assume this
        case Instantiation(generic, items):
            return is_immutable(generic) and all(is_immutable(x) for x in items)
        case Literal() as literal:
            if isinstance(literal.value, tuple):
                return literal.ref == "builtins.tuple" and all(
                    is_immutable(x) for x in literal.value
                )
            return True
        case Row(type=value):
            return is_immutable(value)
        case FunctionType() as f:
            if f.side_effect.update[0] is not None:
                return False
            return True
        case Ref(name):
            module, name = name.split(".")
            if module != "builtins":
                return False
            if name in ["list", "dict", "set"]:
                return False
            if name[0].isupper():
                return name == "NoneType"
            return True
        case Access(items, arg):
            return is_immutable(items) and is_immutable(arg)
        case _:
            return False


class TypeCollector:
    def __init__(self):
        self.symtable: SymbolTable | None = None

    @contextlib.contextmanager
    def enter_scope(self, name: str | None) -> None:
        self.symtable = SymbolTable(name, parent=self.symtable)
        yield
        self.symtable = self.symtable.parent

    def expr_to_type(self, expr: ast.expr, default: TypeExpr = ANY) -> TypeExpr:
        if expr is None:
            return default
        return TypeExpressionParser(self.symtable).visit(expr)

    def visit_ClassDef(self, cdef: ast.ClassDef) -> Class | Module | Instantiation:
        with self.enter_scope(cdef.name):
            type_params = self.symtable.add_generics(*cdef.type_params)
            base_classes_list = []
            protocol = False
            for base in cdef.bases:
                match base:
                    case ast.Name(id="Protocol"):
                        protocol = True
                    case ast.Subscript(
                        value=ast.Name(id=("Protocol" | "Generic") as id),
                    ):
                        protocol |= id == "Protocol"
                    case ast.Name(id=id):
                        base_classes_list.append(Ref(id))
                    case _:
                        raise NotImplementedError(f"{base!r}")

            name_decorators = {
                decorator.id: decorator
                for decorator in cdef.decorator_list
                if isinstance(decorator, ast.Name)
            }
            if "module" in name_decorators:
                module_dict = typed_dict(
                    [
                        row
                        for index, stmt in enumerate(cdef.body)
                        for row in self.stmt_to_rows(stmt, index)
                    ]
                )
                return Module(f"{self.symtable.scope_name}.{cdef.name}", module_dict)
            else:
                class_dict = typed_dict(
                    [
                        infer_self(row)
                        for index, stmt in enumerate(cdef.body)
                        for row in self.stmt_to_rows(stmt, index)
                    ]
                )

                res = Class(
                    cdef.name,
                    class_dict,
                    inherits=tuple(base_classes_list),
                    protocol=protocol,
                    type_params=type_params,
                )
                typetype = Instantiation(Ref("builtins.type"), (res,))
                _discard(
                    metaclass=Class(
                        f"__{cdef.name}_metaclass__",
                        typed_dict(
                            [
                                make_row(
                                    0,
                                    "__call__",
                                    FunctionType(
                                        typed_dict(
                                            [
                                                make_row(0, "cls", TypeVar("Infer")),
                                            ]
                                        ),
                                        Ref("type"),
                                        side_effect=SideEffect(new=True),
                                        is_property=False,
                                        type_params=(),
                                    ),
                                ),
                            ]
                        ),
                        inherits=(Ref("builtins.type"),),
                        protocol=False,
                        type_params=type_params,
                    )
                )
                return typetype

    def visit_FunctionDef(self, fdef: ast.FunctionDef) -> FunctionType:
        with self.enter_scope(fdef.name):
            type_params = self.symtable.add_generics(*fdef.type_params)

            returns = self.expr_to_type(fdef.returns)
            name_decorators = {
                decorator.id: decorator
                for decorator in fdef.decorator_list
                if isinstance(decorator, ast.Name)
            }
            call_decorators = {
                decorator.func.id: decorator
                for decorator in fdef.decorator_list
                if isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
            }
            # Parse @alias[self._buffer] decorators
            alias_fields: tuple[str, ...] = ()
            for decorator in fdef.decorator_list:
                if (
                    isinstance(decorator, ast.Subscript)
                    and isinstance(decorator.value, ast.Name)
                    and decorator.value.id == "alias"
                    and isinstance(decorator.slice, ast.Attribute)
                    and isinstance(decorator.slice.value, ast.Name)
                    and decorator.slice.value.id == "self"
                ):
                    alias_fields = alias_fields + (decorator.slice.attr,)
            # Parse @accessor(self[index]) decorator - returns objects from self's * field
            is_accessor = False
            for decorator in fdef.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "accessor"
                    and len(decorator.args) == 1
                    and isinstance(decorator.args[0], ast.Subscript)
                    and isinstance(decorator.args[0].value, ast.Name)
                    and decorator.args[0].value.id == "self"
                ):
                    is_accessor = True
            update = call_decorators.get("update")
            if update is not None:
                assert isinstance(update, ast.Call)
                update_arg = update.args[0]
                if isinstance(update_arg, ast.Constant) and isinstance(
                    update_arg.value, str
                ):
                    update_arg = ast.parse(update_arg.s).body[0].value
                update_type = self.expr_to_type(update_arg)
                update_args = tuple(self.expr_to_type(x) for x in update.args[1:])
            else:
                update_type = None
                update_args = ()
            # side_effect = parse_side_effect(fdef.body)
            side_effect = SideEffect(
                new="new" in name_decorators and not is_immutable(returns),
                update=(update_type, update_args),
                points_to_args="points_to_args" in name_decorators,
                alias=alias_fields,
                accessor=is_accessor,
            )
            is_property = "property" in name_decorators

            params = typed_dict(
                [
                    make_row(
                        index, arg.arg, self.expr_to_type(arg.annotation, default=TOP)
                    )
                    for index, arg in enumerate(fdef.args.args)
                ]
            )

            # Trick: add free generic vars to type_params
            free_generic_vars = {
                t
                for node in fdef.args.args
                for x in free_vars(node)
                if isinstance(t := self.symtable.lookup(x), TypeVar)
                if t not in type_params
            }
            type_params = (*type_params, *free_generic_vars)

            f = FunctionType(
                params=params,
                return_type=returns,
                side_effect=side_effect,
                is_property=is_property,
                type_params=type_params,
            )
            return f

    def stmt_to_rows(self, definition: ast.stmt, index: int) -> typing.Iterator[Row]:
        match definition:
            case ast.Pass():
                return
            case ast.AnnAssign(target=ast.Name(id=name), annotation=annotation):
                yield make_row(index, name, self.expr_to_type(annotation))
            case ast.Import(names=aliases):
                for alias in aliases:
                    asname = alias.asname or alias.name
                    yield make_row(index, asname, Ref(alias.name))
            case ast.ImportFrom(module=module, names=aliases):
                for alias in aliases:
                    asname = alias.asname or alias.name
                    yield make_row(index, asname, Ref(f"{module}.{alias.name}"))
            case ast.ClassDef() as cdef:
                t = self.visit_ClassDef(cdef)
                yield make_row(index, cdef.name, t)
            case ast.FunctionDef() as fdef:
                t = self.visit_FunctionDef(fdef)
                yield make_row(index, fdef.name, t)
            case ast.If():
                return
            case ast.Expr():
                return []
            case _:
                raise NotImplementedError(f"{definition!r}, {type(definition)}")

    def visit_Module(self, module: ast.Module) -> Module:
        self.symtable.names = {
            node.name
            for node in module.body
            if isinstance(node, (ast.ClassDef, ast.FunctionType))
        }
        self.symtable.aliases = {
            name.asname or name.name: name.name
            for node in module.body
            if isinstance(node, ast.Import)
            for name in node.names
        }
        for node in module.body:
            if isinstance(node, ast.ImportFrom):
                for name in node.names:
                    asname = name.asname or name.name
                    self.symtable.aliases[asname] = f"{node.module}.{name.name}"

        class_dict = typed_dict(
            [
                row
                for index, stmt in enumerate(module.body)
                if not isinstance(stmt, ast.Assign)
                for row in self.stmt_to_rows(stmt, index)
            ]
        )
        return Module(self.symtable.scope_name, class_dict)


def parse_file(path: pathlib.Path) -> Module:
    tree = ast.parse(path.read_text())
    module = TypeCollector()
    with module.enter_scope(Path(path).stem):
        return module.visit_Module(tree)


TYPESHED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../typeshed_mini")
)
MODULES = Module(
    "typeshed",
    typed_dict(
        [
            make_row(
                index,
                file.split(".")[0],
                parse_file(pathlib.Path(f"{TYPESHED_DIR}/{file}")),
            )
            for index, file in enumerate(os.listdir(TYPESHED_DIR))
        ]
    ),
)


def main() -> None:
    pretty_print_type(MODULES)


def is_bound_method(t: TypeExpr) -> bool:
    return (
        isinstance(t, FunctionType)
        and t.side_effect.bound_method
        or isinstance(t, Overloaded)
        and any(is_bound_method(x) for x in t.items)
    )


def get_side_effect(applied: Overloaded) -> SideEffect:
    # Aggregate alias fields from all overloads
    all_alias: tuple[str, ...] = ()
    for x in applied.items:
        for field in x.side_effect.alias:
            if field not in all_alias:
                all_alias = all_alias + (field,)
    return SideEffect(
        new=any(x.side_effect.new for x in applied.items),
        update=(
            join_all(x.side_effect.update[0] for x in applied.items),
            applied.items[0].side_effect.update[1],
        ),
        bound_method=any(is_bound_method(x) for x in applied.items),
        points_to_args=any(x.side_effect.points_to_args for x in applied.items),
        alias=all_alias,
        accessor=any(x.side_effect.accessor for x in applied.items),
    )


def iter_children(t: TypeExpr) -> typing.Iterator[TypeExpr]:
    yield t
    match t:
        case TypeVar():
            pass
        case Ref():
            pass
        case Literal() as x:
            yield from iter_children(x.ref)
            if isinstance(x.value, tuple):
                for item in x.value:
                    yield from iter_children(item)
        case Overloaded() as x:
            yield from iter_children(x.items)
        case Union() as x:
            yield from iter_children(x.items)
        case TypedDict() as x:
            yield from iter_children(x.items)
        case Instantiation() as x:
            yield from iter_children(x.type_params)
            yield from iter_children(x.generic)
        case Row() as x:
            yield from iter_children(x.type)
        case FunctionType() as x:
            yield from iter_children(x.params)
            yield from iter_children(x.return_type)
            yield from iter_children(x.side_effect)
        case _:
            raise NotImplementedError(t)


def is_monomorphized(t: TypeExpr) -> bool:
    for x in iter_children(t):
        if isinstance(x, Literal) and isinstance(x.value, tuple):
            # Special case for tuple and lists
            return True
        if isinstance(x, Instantiation):
            if any(not isinstance(type_param, TypeVar) for type_param in x.type_args):
                return True
    return False
