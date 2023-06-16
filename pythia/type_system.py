from __future__ import annotations as _

import ast
import enum
import os
import typing
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class TypeExpr:
    pass


T = typing.TypeVar('T', bound=TypeExpr)
K = typing.TypeVar('K')


@dataclass(frozen=True)
class Ref(TypeExpr):
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TypeVar(TypeExpr):
    name: str
    is_args: bool = False

    def __repr__(self) -> str:
        return f'*{self.name}' if self.is_args else self.name


@dataclass(frozen=True)
class Star(TypeExpr):
    items: tuple[TypeExpr, ...]

    def __repr__(self) -> str:
        return f'*{self.items}'


@dataclass(frozen=True)
class Index:
    number: typing.Optional[int]
    name: typing.Optional[str]

    def __repr__(self) -> str:
        if self.number is None:
            if self.name is None:
                return 'None'
            return f'{self.name}'
        if self.name is None:
            return f'{self.number}'
        return f'({self.number}){self.name}'

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


@dataclass(frozen=True)
class Union(TypeExpr):
    items: frozenset[TypeExpr]

    def __repr__(self) -> str:
        if not self.items:
            return 'BOT'
        return f'{{{" | ".join(f"{item}" for item in self.items)}}}'

    def squeeze(self) -> TypeExpr:
        if len(self.items) == 1:
            return next(iter(self.items))
        return self


@dataclass(frozen=True)
class Row(TypeExpr):
    # an index has either a name or an index or both
    index: Index
    type: TypeExpr

    def __repr__(self) -> str:
        return f'{self.index}: {self.type}'

    def __le__(self, other: Row) -> bool:
        return self.index <= other.index

    def __lt__(self, other: Row) -> bool:
        return self.index < other.index


def make_row(number: typing.Optional[int], name: typing.Optional[str], t: TypeExpr) -> Row:
    return Row(Index(number if number is not None else None,
                     name if name is not None else None), t)


@dataclass(frozen=True)
class Literal(TypeExpr):
    value: int | str | bool | float | tuple | TypeVar | None
    ref: Ref

    def __repr__(self) -> str:
        if isinstance(self.value, tuple):
            return f'Literal({list(self.value)})'
        return f'Literal({self.value!r})'


def literal(value: int | str | bool | float | tuple | list | TypeVar | None) -> Literal:
    match value:
        case int(): ref = Ref('builtins.int')
        case float(): ref = Ref('builtins.float')
        case str(): ref = Ref('builtins.str')
        case bool(): ref = Ref('builtins.bool')
        case None: ref = Ref('builtins.NoneType')
        case tuple(): ref = Ref('builtins.tuple')
        case list():
            value = tuple(value)
            ref = Ref('builtins.list')
        case _:
            assert False, f'Unknown literal type {value!r}'
    return Literal(value, ref)


@dataclass(frozen=True)
class TypedDict(TypeExpr):
    items: frozenset[Row]

    def __repr__(self) -> str:
        items = self.items
        # if not items:
        #     return 'TOP'
        if all(isinstance(item, Row) for item in items):
            res = ", ".join(f"{item.index}={item.type}"
                            for item in sorted(self.row_items(), key=lambda item: item.index))
            if len(items) == 1:
                return f'({res},)'
            return f'({res})'
        return f'{{{" & ".join(f"{decl}" for decl in items)}}}'

    def row_items(self) -> frozenset[Row]:
        assert all(isinstance(item, Row) for item in self.items)
        return typing.cast(frozenset[Row], self.items)

    def split_by_index(self: TypedDict, value: int | str) -> tuple[TypedDict, TypedDict]:
        index = index_from_literal(value)
        items = self.row_items()
        good = typed_dict([item for item in items if match_index(item.index, index)])
        bad = typed_dict([item for item in items if not match_index(item.index, index)])
        return good, bad

    def squeeze(self: TypedDict) -> TypedDict | Row:
        if len(self.items) == 1:
            return next(iter(self.items))
        return self


@dataclass(frozen=True)
class Overloaded(TypeExpr):
    items: tuple[FunctionType, ...]

    def __repr__(self) -> str:
        return f'Overloaded({self.items})'


@dataclass(frozen=True)
class Class(TypeExpr):
    name: str
    class_dict: TypedDict
    inherits: tuple[TypeExpr, ...]
    protocol: bool
    type_params: tuple[TypeVar, ...]

    def __repr__(self) -> str:
        return f'{self.name}'


@dataclass(frozen=True)
class Module(TypeExpr):
    name: str
    class_dict: TypedDict

    def __repr__(self) -> str:
        return f'module {self.name}({self.class_dict})'


@dataclass(frozen=True)
class SideEffect(TypeExpr):
    new: bool
    bound_method: bool = False
    update: typing.Optional[TypeExpr] = None
    points_to_args: bool = False


@dataclass(frozen=True)
class FunctionType(TypeExpr):
    params: TypedDict
    return_type: TypeExpr
    side_effect: SideEffect
    is_property: bool
    type_params: tuple[TypeVar, ...]

    def __repr__(self) -> str:
        if self.is_property == literal(True):
            return f'->{self.return_type}'
        else:
            type_params = ', '.join(str(x) for x in self.type_params)
            new = "new " if self.new() else ""
            update = "{update " + str(self.side_effect.update) + "}@" if self.side_effect.update else ""
            return f'[{type_params}]({self.params} -> {update}{new}{self.return_type})'

    def new(self):
        return self.side_effect.new


@dataclass(frozen=True)
class Instantiation(TypeExpr):
    generic: Class | FunctionType | Ref
    type_args: tuple[TypeExpr, ...]

    def __repr__(self) -> str:
        return f'{self.generic}[{", ".join(repr(x) for x in self.type_args)}]'


@dataclass(frozen=True)
class Access(TypeExpr):
    items: TypeExpr  # Typevar before resolution
    arg: TypeExpr

    def __repr__(self) -> str:
        return f'{self.items}.[{self.arg}]'


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
                return replace(t,
                               value=unpack_star(bind_typevars(x, context) for x in t.value))
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
            return replace(f,
                           params=new_params,
                           return_type=new_return_type,
                           side_effect=new_side_effect)
        case Union(items):
            return join_all(bind_typevars(item, context) for item in items)
        case Row() as row:
            return replace(row,
                           type=bind_typevars(row.type, context))
        case SideEffect() as s:
            if s.update is None:
                return s
            return replace(s,
                           update=bind_typevars(s.update, context))
        case Class(type_params=type_params, class_dict=class_dict, inherits=inherits) as klass:
            context = {k: v for k, v in context.items() if k not in type_params}
            if not context:
                return klass
            return replace(klass,
                           class_dict=bind_typevars(class_dict, context),
                           inherits=tuple(bind_typevars(x, context) for x in inherits))
        case Instantiation(generic, type_args) as instantiation:
            new_type_args = unpack_star(bind_typevars(x, context) for x in type_args)
            return replace(instantiation,
                           generic=bind_typevars(generic, context),
                           type_args=new_type_args)
        case Access(argsvar, arg):
            choices = bind_typevars(argsvar, context)
            actual_arg = bind_typevars(arg, context)
            if isinstance(choices, Star) and isinstance(actual_arg, Literal) and isinstance(actual_arg.value, int):
                return choices.items[actual_arg.value]
            return Access(choices, actual_arg)
        case SideEffect() as s:
            if s.update is None:
                return s
            return replace(s,
                           update=bind_typevars(s.update, context))
    raise NotImplementedError(f'{t!r}, {type(t)}')


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
        else:
            assert isinstance(f, FunctionType), f
            collect.append(f)
    # assert len(collect) > 0
    return Overloaded(tuple(collect))


def union(items: typing.Iterable[TypeExpr], squeeze=True) -> TypeExpr:
    items = frozenset(items)
    if any(isinstance(x, Literal) for x in items):
        if any(isinstance(x, Ref) for x in items):
            pass
    # assert len(items) > 0
    res = Union(items)
    if squeeze:
        return res.squeeze()
    return res


TOP = typed_dict([])
BOTTOM = Union(frozenset())
ANY = Ref('typing.Any')


def join(t1: TypeExpr, t2: TypeExpr) -> TypeExpr:
    if t1 == t2:
        return t1
    if t1 == BOTTOM:
        return t2
    if t2 == BOTTOM:
        return t1
    if t1 == TOP or t2 == TOP:
        return TOP
    match t1, t2:
        case Union(items1), Union(items2):
            return Union(items1 | items2).squeeze()
        case (Union(items), other) | (other, Union(items)):  # type: ignore
            return Union(items | {other}).squeeze()
        case (Overloaded() as f, Overloaded() as g):
            if len(f.items) == 1 and len(g.items) == 1:
                res = join(f.items[0], g.items[0])
                if isinstance(res, FunctionType):
                    return overload([res])
            return union([t1, t2])
        case (Overloaded(), _) | (_, Overloaded()):
            return TOP
        case (FunctionType() as f1, FunctionType() as f2):
            if (f1.is_property, f1.type_params) == (f2.is_property, f2.type_params):
                # TODO: check that f1.params and f2.params are compatible
                new_params = meet(f1.params, f2.params)
                if isinstance(new_params, Row):
                    new_params = typed_dict([new_params])
                assert isinstance(new_params, TypedDict)
                return replace(f1,
                               params=new_params,
                               return_type=join(f1.return_type, f2.return_type),
                               side_effect=join(f1.side_effect, f2.side_effect))
            return union([f1, f2])
        case (Literal() as l1, Literal() as l2):
            if l1.ref == l2.ref:
                if isinstance(l1.value, tuple):
                    if len(l1.value) == len(l2.value):
                        return Literal(tuple(join(t1, t2) for t1, t2 in zip(l1.value, l2.value)), l1.ref)
                    if l1.ref == Ref('builtins.list'):
                        return Instantiation(l1.ref, (join_all([*l1.value, *l2.value]),))
                return l1.ref
            assert l1 != l2
            return union([t1, t2])
        case (Literal() as n, Ref() as ref) | (Ref() as ref, Literal() as n) if n.ref == ref:
            return ref
        case (Literal(tuple() as value, ref=ref), Instantiation() as inst) | (Instantiation() as inst, Literal(tuple() as value, ref=ref)) if ref == inst.generic:
            if ref.name == 'builtins.list':
                value = (join_all(value),)
            return join(inst, Instantiation(ref, value))
        case (TypedDict(items1), TypedDict(items2)):  # type: ignore
            return TypedDict(items1.intersection(items2))
        case (Ref() as ref, other) | (other, Ref() as ref):  # type: ignore
            if resolve_static_ref(ref) == other:
                return ref
            return union([ref, other])
        case (Instantiation(generic1, type_args1), Instantiation(generic2, type_args2)) if generic1 == generic2:
            return Instantiation(generic1, tuple(join(t1, t2) for t1, t2 in zip(type_args1, type_args2)))
        case (Row(index1, t1), Row(index2, t2)):
            if index1 == index2:
                return Row(index1, join(t1, t2))
            return BOTTOM
        case (Row(_, t1), (Instantiation(Ref('builtins.list'), type_args))) | (Instantiation(Ref('builtins.list'), type_args), Row(_, t1)):
            return Instantiation(Ref('builtins.list'), tuple(join(t1, t) for t in type_args))
        case (Row(_, t1), (Instantiation(Ref('builtins.tuple'), type_args))) | (Instantiation(Ref('builtins.tuple'), type_args), Row(_, t1)):
            # not exact; should only join at the index of the row
            return Instantiation(Ref('builtins.tuple'), tuple(join(t1, t) for t in type_args))
        case Class(), Class():
            return TOP
        case (Class(name="int") | Ref('builtins.int') as c, Literal(int())) | (Literal(int()), Class(name="int") | Ref('builtins.int') as c):
            return c
        case (SideEffect() as s1, SideEffect() as s2):
            return SideEffect(
                new=s1.new | s2.new,
                bound_method=s1.bound_method | s2.bound_method,
                update=join(s1.update, s2.update),
                points_to_args=s1.points_to_args | s2.points_to_args,
            )
        case x, y:
            return union([x, y])
    raise NotImplementedError(f'{t1!r}, {t2!r}')


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
                        return Literal(tuple(meet(t1, t2) for t1, t2 in zip(l1.value, l2.value)), l1.ref)
                    if l1.ref == Ref('builtins.list'):
                        return Instantiation(l1.ref, (meet_all([*l1.value, *l2.value]),))
                return l1.ref
            assert l1 != l2
            assert False, f'{l1!r}, {l2!r}'
        case (Literal(tuple() as value, ref=ref), Instantiation() as inst) | (Instantiation() as inst, Literal(tuple() as value, ref=ref)) if ref == inst.generic:
            if ref.name == 'builtins.list':
                value = (meet_all(value),)
            return meet(inst, Instantiation(ref, value))
        case (Overloaded(), _):
            return TOP
        case (TypedDict(items1), TypedDict(items2)):  # type: ignore
            return typed_dict(items1 | items2)
        case (Ref() as ref, Literal() as n) | (Literal() as n, Ref() as ref) if n.ref == ref:
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
            # if (f1.is_property, f1.side_effect, f1.type_params, f1.return_type) == (f2.is_property, f2.side_effect, f2.type_params, f1.return_type):
            #     new_params = join(f1.params, f2.params)
            #     if not isinstance(new_params, Union):
            #         if isinstance(new_params, Row):
            #             new_params = typed_dict([new_params])
            #         # TODO: side_effect = join(f1.side_effect, f2.side_effect)
            #         assert isinstance(new_params, TypedDict)
            #         return replace(f1, params=new_params,
            #                        return_type=meet(f1.return_type, f2.return_type))
            return overload([f1, f2])
        case (t1, t2):
            raise NotImplementedError(f'{t1!r}, {t2!r}')
    raise AssertionError


def meet_all(items: typing.Iterable[TypeExpr]) -> TypeExpr:
    res: TypeExpr = TOP
    for t in unpack_star(items):
        res = meet(res, t)
    return res


class Action(enum.Enum):
    INDEX = enum.auto()
    SELECT = enum.auto()


def is_subtype(left: TypeExpr, right: TypeExpr) -> bool:
    if right == ANY:
        return True
    if left == ANY:
        return True
    return join(left, right) == right


def index_from_literal(index: int | str) -> Index:
    return Index(index, None) if isinstance(index, int) else Index(None, index)


def match_index(param: Index, arg: Index) -> bool:
    return arg.number == param.number and arg.number is not None or arg.name == param.name and arg.name is not None


def unify_argument(type_params: tuple[TypeVar, ...], param: TypeExpr, arg: TypeExpr) -> typing.Optional[dict[TypeVar, TypeExpr]]:
    if param == arg:
        return {}
    if arg == BOTTOM:
        return {x: BOTTOM for x in (set(type_params) & free_vars_expr(param))}
    if param == TOP:
        return {}
    match param, arg:
        case (Ref('typing.Any'), _) | (_, Ref('typing.Any')):
            return {}
        case Union(items), arg:
            res = [unified for item in items if (unified := unify_argument(type_params, item, arg)) is not None]
            if not res:
                return None
            return {k: join_all(t.get(k, BOTTOM) for t in res) for k in type_params}
        case param, TypedDict(items):
            res = [unified for item in items if (unified := unify_argument(type_params, param, item)) is not None]
            if not res:
                return None
            return {k: join_all(t.get(k, BOTTOM) for t in res) for k in type_params}
        case TypeVar() as param, arg:
            return {param: arg}
        case Class() as param, Instantiation(Class() as arg, arg_args):
            if param.name == arg.name:
                return {k: v for k, v in zip(param.type_params, arg_args)}
            return None
        case Instantiation() as param, Instantiation() as arg:
            if param.generic != arg.generic:
                return None
            ps = typed_dict([make_row(i, None, p) for i, p in enumerate(param.type_args)])
            args = typed_dict([make_row(i, None, p) for i, p in enumerate(arg.type_args)])
            mid_context = unify(type_params, ps, args)
            assert mid_context is not None
            return mid_context.bound_typevars
        case Instantiation() as param, Literal(tuple() as value, ref):
            if param.generic != ref:
                return None
            if ref.name == 'builtins.list':
                value = (join_all(value),)
            return unify_argument(type_params, param, Instantiation(ref, value))
        case param, Instantiation(Ref() as param_type, param_args) if not isinstance(param, Ref):
            return unify_argument(type_params, param, Instantiation(resolve_static_ref(param_type), param_args))
        case Literal() as param, Literal() as arg:
            if param == arg:
                return {}
            if param.ref == arg.ref and isinstance(param.value, tuple):
                assert isinstance(arg.value, tuple)
                mid_context = unify(type_params, typed_dict([make_row(i, None, p) for i, p in enumerate(param.value)]),
                                                 typed_dict([make_row(i, None, p) for i, p in enumerate(arg.value)]))
                assert mid_context is not None
                return mid_context.bound_typevars
            return None
        case Row() as param, Row() as arg:
            if match_index(param.index, arg.index):
                return unify_argument(type_params, param.type, arg.type)
            return None
        case (Class('int') | Ref('builtins.int'), Literal(int())) | (Literal(int()), Class('int') | Ref('builtins.int')):
            return {}
        case Class(), Literal():
            return None
        case param, Ref() as arg:
            return unify_argument(type_params, param, resolve_static_ref(arg))
        case Ref() as param, Literal(ref=arg) if arg == param:
            return {}
        case Ref() as param, arg:
            return unify_argument(type_params, resolve_static_ref(param), arg)
        case param, arg:
            return None
            raise NotImplementedError(f'{param!r}, {arg!r}')
    assert False


@dataclass
class Binding:
    bound_typevars: dict[TypeVar, TypeExpr]
    bound_params: dict[Row, Row]
    unbound_params: set[Row]


def unify(type_params: tuple[TypeVar, ...], params: TypedDict, args: TypedDict) -> typing.Optional[Binding]:
    bound_types: dict[TypeVar, TypeExpr] = {}
    bound_params: dict[Row, Row] = {}
    unbound_params: set[Row] = set(params.row_items())
    unbound_args = set(args.row_items())
    varparam: typing.Optional[Row] = None
    for param in sorted(params.row_items()):
        if isinstance(param.type, TypeVar) and param.type.is_args:
            varparam = param
            continue
        matching_args = {arg for arg in unbound_args if match_index(param.index, arg.index)}
        if not matching_args:
            continue
        unbound_params.remove(param)
        unbound_args -= matching_args
        assert len(matching_args) == 1, f'{matching_args!r}'
        matching_arg = matching_args.pop()
        binding = unify_argument(type_params, param.type, matching_arg.type)
        if binding is None:
            return None
        bound_params[param] = matching_arg
        bindable_type_params = {k for k in type_params if k in binding}
        for k in bindable_type_params:
            if k.is_args:
                items: set[TypeExpr] = {bound_types.get(k, None), binding.get(k, None)} - {None}
                assert len(items) <= 1, f'{items!r}'
                if items:
                    bound_types[k] = items.pop()
            else:
                bound_types[k] = join(bound_types.get(k, BOTTOM),
                                      binding.get(k, BOTTOM))
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
        bound_params={k: bind_typevars(v, bound_types) for k, v in bound_params.items()},
        unbound_params=unbound_params,
    )


def match_row(row: Row, arg: Literal | Ref) -> bool:
    match arg:
        case Literal(value):
            return match_index(row.index, index_from_literal(value))
        case Ref(name):
            return name == 'builtins.int' and row.index.number is not None
    assert False, f'{row!r}, {arg!r}'


def access(t: TypeExpr, arg: TypeExpr) -> Overloaded | Module:
    match t, arg:
        case Class(class_dict=class_dict) | Module(class_dict=class_dict), Literal(str() as value):
            good, bad = class_dict.split_by_index(value)
            types: list[TypeExpr] = [x.type for x in good.row_items()]
            if len(types) == 1:
                [t] = types
                if isinstance(t, Module):
                    return t
                if not isinstance(t, (Overloaded, FunctionType)):
                    return t
            if not types and isinstance(t, Module) and t.name not in ['builtins', 'numpy']:
                return TOP
            result = overload(types)
            assert free_vars_expr(result) == set(), f'{result!r}'
            return result
        case Class() as t, arg:
            getter = access(t, literal('__getitem__'))
            getter_as_property = partial(getter, typed_dict([make_row(1, None, arg)]))
            assert free_vars_expr(getter_as_property) == set(), f'{getter_as_property!r}'
            if getter_as_property == BOTTOM:
                return BOTTOM
            if isinstance(getter_as_property, Union):
                assert len(getter_as_property.items) == 1, f'{getter_as_property!r}'
                [getter_as_property] = getter_as_property.items
            assert isinstance(getter_as_property, Overloaded), f'{getter_as_property!r}'
            return overload([replace(g, is_property=True) for g in getter_as_property.items])
        case Module() as t, arg:
            return access(t.class_dict, arg)
        case Ref() as ref, arg:
            return access(resolve_static_ref(ref), arg)
        case t, TypedDict(items):
            assert False, f'{t!r}, {arg!r}'
        case t, Overloaded(items):
            assert False, f'{t!r}, {arg!r}'
        case Overloaded(tuple()), arg:
            return TOP
        case Union(items), arg:
            return join_all(access(item, arg) for item in items)
        case TypedDict(items), arg:
            # intersect and not meet in order to differentiate between
            # a non-existent attribute and an attribute with type TOP
            return overload([row.type for row in items
                             if match_row(row, arg)])
        case Literal(ref=ref), arg:
            return access(ref, arg)
        case t, Union(items):
            return overload([access(t, item) for item in items])
        case Instantiation() as t, arg:
            # ignore type arguments for now
            return access(t.generic, arg)
        case _:
            raise NotImplementedError(f'{t=}, {arg=}, {type(t)=}, {type(arg)=}')


def subtract_indices(unbound_params: TypedDict, bound_params: TypedDict) -> TypedDict:
    # Fix: subtract index from all indexable params
    # For this, we need to go from lowest to highest,
    # and subtract from it the number of lower-indexed params that became bound
    # (i.e. the number of params that were removed from the row)
    # Here we assume that the params are consecutive, starting from 0
    indexes = [item.index for item in bound_params.row_items()
               if item.index.number is not None]

    if indexes:
        max_bound = max(indexes)
        unbound_params = typed_dict([replace(item, index=item.index - max_bound - 1) if item.index > max_bound else item
                                    for item in unbound_params.row_items()])
    return unbound_params


def split_params_by_arg_indices(params: TypedDict, args: TypedDict) -> tuple[list[tuple[Row, Row]], TypedDict]:
    bound_params: list[tuple[Row, Row]] = []
    unbound_params: list[Row] = []
    arg_items = set(args.row_items())
    for param in params.row_items():
        for arg in arg_items:
            if match_index(param.index, arg.index):
                bound_params.append((param, arg))
                arg_items.remove(arg)
                break
        else:
            unbound_params.append(param)
    return bound_params, typed_dict(unbound_params)


def union_all(iterable: typing.Iterable[set]) -> set:
    result = set()
    for item in iterable:
        result |= item
    return result


def free_vars_expr(t: TypeExpr) -> set[TypeVar]:
    match t:
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
            return free_vars_expr(generic) | union_all(free_vars_expr(arg) for arg in args)
        case TypedDict() as t:
            return free_vars_typed_dict(t)
        case FunctionType() as f:
            result = free_vars_expr(f.return_type) | free_vars_expr(f.params)
            result -= set(f.type_params) & free_vars_expr(f.params)
            return result
        case Class(): return set()
        case Module(): return set()
        case Overloaded() as t:
            return union_all(free_vars_expr(item) for item in t.items)
        case _: assert False, f'{t!r} {type(t)!r}'


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


def partial(callable: TypeExpr, args: TypedDict) -> TypeExpr:
    if callable == TOP:
        return TOP
    if callable == BOTTOM:
        return BOTTOM
    match callable:
        case Overloaded(items):
            # This returns a union of cases, so defaults to TOP.
            applied = []
            skip = set()
            for i, f in enumerate(items):
                if i in skip:
                    continue
                first_bound_params = None
                overloaded = []
                for j, f in enumerate(items[i:], i):
                    if j in skip:
                        continue

                    binding = unify(f.type_params, f.params, args)
                    if binding is None:
                        if first_bound_params is not None:
                            continue
                        else:
                            break
                    if first_bound_params is None:
                        first_bound_params = binding.bound_params
                    if binding.bound_params == first_bound_params:
                        skip.add(j)
                        f = replace(f,
                                    type_params=tuple(v for v in f.type_params if v not in binding.bound_typevars),
                                    params=subtract_indices(typed_dict(binding.unbound_params),
                                                            typed_dict(binding.bound_params.keys())),
                                    return_type=bind_typevars(f.return_type, binding.bound_typevars),
                                    side_effect=bind_typevars(f.side_effect, binding.bound_typevars),
                                    )
                        overloaded.append(f)
                del f
                if not overloaded:
                    continue
                applied.append(overload(overloaded))

                if first_bound_params is None:
                    continue

                args = typed_dict(
                    replace(arg, type=subtract_type_underapprox(arg.type, param.type))
                    for param, arg in first_bound_params.items()
                )
                if any(arg.type == BOTTOM for arg in args.row_items()):
                    break
            return join_all(applied)
        case FunctionType() as f:
            binding = unify(f.type_params, f.params, args)
            # This returns a single case, so defaults to BOTTOM.
            if binding is None:
                return BOTTOM
            return replace(f,
                           type_params=tuple(v for v in f.type_params if v not in binding.bound_typevars),
                           params=subtract_indices(typed_dict(binding.unbound_params),
                                                   typed_dict(binding.bound_params.keys())),
                           return_type=bind_typevars(f.return_type, binding.bound_typevars),
                           side_effect=bind_typevars(f.side_effect, binding.bound_typevars))
        case Union(items):
            return union([partial(item, args) for item in items])
        case Class(class_dict=class_dict), arg:
            dunder = subscr(class_dict, literal('__call__'))
            return partial(dunder, arg)
        case _:
            assert False, f'Cannot call {callable} with {args}'


def partial_positional(f: TypeExpr, args: tuple[TypeExpr, ...]) -> TypeExpr:
    return partial(f, typed_dict([make_row(i, None, arg) for i, arg in enumerate(args)]))


def bind_self_function(f: FunctionType, selftype: TypeExpr) -> FunctionType:
    res = partial(f, typed_dict([make_row(0, 'self', selftype)]))
    if isinstance(res, FunctionType):
        res = replace(res, side_effect=replace(res.side_effect, bound_method=True))
    elif isinstance(res, Overloaded):
        res = overload(replace(item, side_effect=replace(item.side_effect, bound_method=True))
                       for item in res.items)
    return res


def bind_self(attr: Overloaded, selftype: TypeExpr) -> Overloaded:
    if isinstance(selftype, Module):
        return attr
    if isinstance(selftype, Ref) and isinstance(resolve_static_ref(selftype), Module):
        return attr
    match attr:
        case Overloaded(items):
            return overload([bind_self_function(item, selftype) for item in items])
        case FunctionType() as attr:
            return overload([bind_self_function(attr, selftype)])
        case Union(items):
            return join_all(bind_self(item, selftype) for item in items)
        case _:
            return attr


def get_return(callable: TypeExpr) -> TypeExpr:
    if callable == TOP:
        return TOP
    if callable == BOTTOM:
        return BOTTOM
    match callable:
        case Overloaded(items):
            return meet_all(get_return(item) for item in items)
        case Union(items):
            return join_all(get_return(item) for item in items)
        case FunctionType(return_type=return_type):
            return return_type
    assert False, f'{callable!r}'


def subscr(selftype: TypeExpr, index: TypeExpr) -> TypeExpr:
    if selftype == TOP:
        return TOP
    if selftype == BOTTOM:
        return BOTTOM
    attr_type = access(selftype, index)
    if attr_type == TOP:
        # non-existent attribute
        return BOTTOM
    result = bind_self(attr_type, selftype)
    if isinstance(result, FunctionType):
        result = overload([result])
    if isinstance(result, Overloaded):
        if any(f.is_property for f in result.items):
            assert all(f.is_property for f in result.items)
            return union(f.return_type for f in result.items)
    assert not free_vars_expr(result), f'{result!r}'
    return result


def call(callable: TypeExpr, args: TypedDict) -> TypeExpr:
    resolved = partial(callable, args)
    result = get_return(resolved)
    assert not free_vars_expr(result), f'{result!r}'
    return result


def make_list_constructor() -> Overloaded:
    args = TypeVar('Args', is_args=True)
    return_type = literal([args])
    return overload([FunctionType(params=typed_dict([make_row(0, 'self', args)]),
                                  return_type=return_type,
                                  side_effect=SideEffect(new=True, points_to_args=True),
                                  is_property=False,
                                  type_params=(args,))])


def make_tuple_constructor() -> Overloaded:
    args = TypeVar('Args', is_args=True)
    return_type = Instantiation(Ref('builtins.tuple'), (args,))
    return overload([FunctionType(params=typed_dict([make_row(0, 'self', args)]),
                                  return_type=return_type,
                                  side_effect=SideEffect(new=True, points_to_args=True),
                                  is_property=False,
                                  type_params=(args,))])


def make_slice_constructor() -> Overloaded:
    return_type = Ref('builtins.slice')
    NONE = literal(None)
    INT = Ref('builtins.int')
    both = union([NONE, INT])
    return overload([FunctionType(params=typed_dict([make_row(0, 'start', both),
                                                     make_row(1, 'end', both)]),
                                  return_type=return_type,
                                  side_effect=SideEffect(new=True),
                                  is_property=False,
                                  type_params=())])


def binop_to_dunder_method(op: str) -> tuple[str, typing.Optional[str]]:
    match op:
        case '+': return '__add__', '__radd__'
        case '-': return '__sub__', '__rsub__'
        case '*': return '__mul__', '__rmul__'
        case '/': return '__truediv__', '__rtruediv__'
        case '/': return '__truediv__', '__rtruediv__'
        case '//': return '__floordiv__', '__rfloordiv__'
        case '%': return '__mod__', '__rmod__'
        case '**': return '__pow__', '__rpow__'
        case '<<': return '__lshift__', '__rlshift__'
        case '>>': return '__rshift__', '__rrshift__'
        case '&': return '__and__', '__rand__'
        case '|': return '__or__', '__ror__'
        case '^': return '__xor__', '__rxor__'
        case '@': return '__matmul__', '__rmatmul__'
        case '==': return '__eq__', None
        case '!=': return '__ne__', None
        case '<': return '__lt__', None
        case '<=': return '__le__', None
        case '>': return '__gt__', None
        case '>=': return '__ge__', None
        case _: raise NotImplementedError(f'{op!r}')


def unop_to_dunder_method(op: str) -> str:
    match op:
        case '-': return '__neg__'
        case '+': return '__pos__'
        case '~': return '__invert__'
        case 'not': return '__bool__'
        case 'iter': return '__iter__'
        case 'yield iter': return '__iter__'
        case 'next': return '__next__'
        case _: raise NotImplementedError(f'{op!r}')


def get_binop(left: TypeExpr, right: TypeExpr, op: str) -> TypeExpr:
    lop, rop = binop_to_dunder_method(op)
    left_ops = access(left, literal(lop))
    if left_ops == TOP:
        return TOP
    if isinstance(left_ops, FunctionType):
        left_ops = overload([left_ops])
    assert isinstance(left_ops, Overloaded), f'{left_ops!r}'
    if left == right:
        return bind_self(left_ops, left)
    right_ops = access(right, literal(rop))
    if right_ops == TOP:
        return TOP
    if isinstance(right_ops, FunctionType):
        right_ops = overload([right_ops])
    if right_ops != BOTTOM:
        assert isinstance(right_ops, Overloaded), f'{right_ops!r}'
        right_ops = overload(swap_binop_params(rf) for rf in right_ops.items)
        ops = overload([left_ops, right_ops])
    else:
        ops = left_ops
    result = bind_self(ops, left)
    assert not free_vars_expr(result), f'{result!r}'
    return result


def swap_binop_params(rf: FunctionType) -> FunctionType:
    assert len(rf.params.items) == 2
    left, right = sorted(rf.params.items)
    right_ops = replace(rf, params=typed_dict([replace(right, index=index_from_literal(0)),
                                               replace(left, index=index_from_literal(1))]))
    return right_ops


def partial_binop(left: TypeExpr, right: TypeExpr, op: str) -> TypeExpr:
    binop_func = get_binop(left, right, op)
    if binop_func == BOTTOM:
        # assume there is an implementation.
        return TOP
    return partial(binop_func, typed_dict([make_row(0, None, right)]))


def get_unop(left: TypeExpr, op: str) -> TypeExpr:
    return subscr(left, literal(unop_to_dunder_method(op)))


def resolve_static_ref(ref: Ref) -> TypeExpr:
    return resolve_relative_ref(ref, MODULES)


def resolve_relative_ref(ref: Ref, module: Module) -> TypeExpr:
    result: TypeExpr = module
    for attr in ref.name.split('.'):
        result = subscr(result, literal(attr))
    return result


def infer_self(row: Row) -> Row:
    function = row.type
    if not isinstance(function, FunctionType):
        return row
    params = function.params
    if not params:
        raise TypeError(f'Cannot bind self to {function}')
    self_args, other_args = params.split_by_index(0)
    [self_arg_row] = self_args.row_items()
    type_params = function.type_params
    if self_arg_row.type == ANY:
        self_type = TypeVar('Self')
        self_arg_row = replace(self_arg_row, type=self_type)
        type_params = (self_type, *type_params)
    g = replace(function,
                params=typed_dict([self_arg_row, *other_args.row_items()]),
                type_params=type_params)
    return Row(row.index, g)


def pretty_print_type(t: Module | TypeExpr, indent: int = 0) -> None:
    match t:
        case TypedDict(items):
            for row in items:
                pretty_print_type(row, indent)
        case Overloaded(items):
            for row in items:
                pretty_print_type(row, indent)
        case Row(index, typeexpr):
            if index.name is None:
                print(' ' * indent, end='')
            else:
                print(' ' * indent, index.name, '=', end='', sep='')
            if isinstance(typeexpr, (Overloaded, TypedDict)):
                print()
            pretty_print_type(typeexpr, indent)
        case FunctionType(params, return_type, side_effect, is_property, type_params):
            # pretty_params = ', '.join(f'{row.index.name}: {row.type}'
            #                           for row in sorted(params.row_items(), key=lambda x: x.index))
            pretty_type_params = ', '.join(str(x) for x in type_params)
            print(f'[{pretty_type_params}]({params}) -> {"new " if side_effect.new else ""}{return_type}')
        case Class(name, class_dict=class_dict, inherits=inherits, protocol=protocol, type_params=type_params):
            kind = 'protocol' if protocol else 'class'
            pretty_type_params = ', '.join(str(x) for x in type_params)
            print(f'{kind} {name}[{pretty_type_params}]({", ".join(str(x) for x in inherits)})')
            pretty_print_type(class_dict, indent + 4)
        case Module(name, typeexpr):
            print(f'module {name}:', sep='')
            pretty_print_type(typeexpr, indent + 4)
        case Ref(name):
            print(f'{name}')
        case Index(name, number):
            if number is None:
                print(f'{name}')
            elif name is None:
                print(f'{number}')
            else:
                print(f'({number}){name}')
        case Literal(value):
            print(f'{value}', end='')
        case _:
            raise NotImplementedError(f'{t!r}, {type(t)}')


def module_to_type(module: ast.Module, name: str) -> Module:
    def free_vars(node: ast.expr | ast.arg) -> set[str]:
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    def expr_to_type(expr: typing.Optional[ast.expr]) -> TypeExpr:
        match expr:
            case None:
                return ANY
            case ast.Constant(value):
                return literal(value)
            case ast.Name(id=id):
                if id in generic_vars:
                    return generic_vars[id]
                if id in global_aliases:
                    return Ref(global_aliases[id])
                if id in global_names:
                    return Ref(f'{name}.{id}')
                else:
                    return Ref(f'builtins.{id}')
            case ast.Starred(value=ast.Name(id=id)):
                return TypeVar(id, is_args=True)
            case ast.Subscript(value=value, slice=slice):
                generic = expr_to_type(value)
                if isinstance(generic, TypeVar):
                    return Access(generic, expr_to_type(slice))
                match slice:
                    case ast.Tuple(elts=[arg, ast.Constant(value=x)]) if str(x) == 'Ellipsis':
                        raise NotImplementedError(f'{generic}[{expr_to_type(arg)}, ...]')
                    case ast.Tuple(elts=elts):
                        items = tuple(expr_to_type(x) for x in elts)
                    case expr:
                        items = (expr_to_type(expr),)
                return Instantiation(generic, items)
            case ast.Attribute(value=value, attr=attr):
                ref: TypeExpr = expr_to_type(value)
                assert isinstance(ref, Ref), f'Expected Ref, got {ref!r} for {value!r}'
                return Ref(f'{ref.name}.{attr}')
            case ast.BinOp(left=left, op=op, right=right):
                left_type = expr_to_type(left)
                right_type = expr_to_type(right)
                if isinstance(op, ast.BitOr):
                    return union([left_type, right_type])
                raise NotImplementedError(f'{left_type} {op} {right_type}')
            case _:
                raise NotImplementedError(f'{expr!r}')

    def parse_generic_arguments(slice: ast.expr) -> tuple[TypeVar, ...]:
        match slice:
            case ast.Tuple(elts=elts):
                return tuple(y for x in elts for y in parse_generic_arguments(x))
            case ast.Name(id=id):
                return (TypeVar(id),)
            case ast.Starred(value=ast.Name(id=id)):
                return (TypeVar(id, is_args=True),)
            case _:
                raise NotImplementedError(f'{slice!r}')

    # convert a Python ast to a type expression
    def stmt_to_rows(definition: ast.stmt, index: int) -> typing.Iterator[Row]:
        match definition:
            case ast.Pass():
                return
            case ast.AnnAssign(target=ast.Name(id=name), annotation=annotation):
                yield make_row(index, name, expr_to_type(annotation))
            case ast.Import(names=aliases):
                for alias in aliases:
                    asname = alias.asname
                    if asname is None:
                        continue
                    yield make_row(index, asname, Ref(alias.name))
            case ast.ClassDef(name=name, body=body, bases=base_expressions):
                base_classes_list = []
                protocol = False
                type_params: tuple[TypeVar, ...] = ()
                for base in base_expressions:
                    match base:
                        case ast.Name(id='Protocol'):
                            protocol = True
                        case ast.Subscript(value=ast.Name(id=('Protocol' | 'Generic') as id), slice=param_slice):
                            protocol = id == 'Protocol'
                            type_params = parse_generic_arguments(param_slice)
                        case ast.Name(id=id):
                            base_classes_list.append(Ref(id))
                        case _:
                            raise NotImplementedError(f'{base!r}')

                metaclass = Class(f'__{name}_metaclass__',
                                  typed_dict([
                                      make_row(0, '__call__', FunctionType(typed_dict([
                                          make_row(0, 'cls', TypeVar('Infer')),
                                      ]), Ref('type'),
                                          side_effect=SideEffect(new=True),
                                          is_property=False,
                                          type_params=())),
                                  ]),
                                  inherits=(Ref('builtins.type'),),
                                  protocol=False,
                                  type_params=())

                class_dict = typed_dict([infer_self(row) for index, stmt in enumerate(body)
                                         for row in stmt_to_rows(stmt, index)])
                res = Class(name, class_dict, inherits=tuple(base_classes_list), protocol=protocol,
                            type_params=type_params)

                yield make_row(index, name, res)
            case ast.FunctionDef() as fdef:
                freevars = {x for node in fdef.args.args for x in free_vars(node)}
                returns = expr_to_type(fdef.returns)
                name_decorators = {decorator.id: decorator for decorator in fdef.decorator_list if isinstance(decorator, ast.Name)}
                call_decorators = {decorator.func.id: decorator for decorator in fdef.decorator_list if isinstance(decorator, ast.Call)}
                update = call_decorators.get('update')
                if update is not None:
                    assert isinstance(update, ast.Call)
                    assert len(update.args) == 1
                    update_type = expr_to_type(update.args[0])
                else:
                    update_type = None
                side_effect = SideEffect(
                    new='new' in name_decorators and not is_immutable(returns),
                    update=update_type,
                    points_to_args='points_to_args' in name_decorators,
                )
                is_property = 'property' in name_decorators
                type_params = tuple(generic_vars[x] for x in freevars if x in generic_vars)
                params = typed_dict(
                    [make_row(index, arg.arg, expr_to_type(arg.annotation))
                     for index, arg in enumerate(fdef.args.args)]
                )
                f = FunctionType(params=params,
                                 return_type=returns,
                                 side_effect=side_effect,
                                 is_property=is_property,
                                 type_params=type_params)

                yield make_row(index, fdef.name, f)
            case ast.If():
                return
            case ast.Expr():
                return []
            case _:
                raise NotImplementedError(f'{definition!r}, {type(definition)}')

    global_names = {node.name for node in module.body if isinstance(node, (ast.ClassDef, ast.FunctionType))}
    generic_vars = {node.targets[0].id: TypeVar(node.targets[0].id, node.value.func.id == 'TypeVarTuple')
                    for node in module.body
                        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name) and node.value.func.id in ['TypeVar', 'TypeVarTuple']
                        and isinstance(node.targets[0], ast.Name)}
    global_aliases = {name.asname: name.name
                      for node in module.body if isinstance(node, ast.Import)
                      for name in node.names if name.asname is not None}
    class_dict = typed_dict([row for index, stmt in enumerate(module.body)
                             if not isinstance(stmt, ast.Assign)
                             for row in stmt_to_rows(stmt, index)])
    return Module(name, class_dict)


def is_immutable(value: TypeExpr) -> bool:
    match value:
        case Overloaded(items) | TypedDict(items) | Union(items):
            return all(is_immutable(x) for x in items)
        case Instantiation(generic, items):
            return is_immutable(generic) and all(is_immutable(x) for x in items)
        case Literal():
            return True
        case Row(type=value):
            return is_immutable(value)
        case FunctionType():
            return True
        case Ref(name):
            module, name = name.split('.')
            if module != 'builtins':
                return False
            if name in ['list', 'dict', 'set']:
                return False
            if name[0].isupper():
                return name == 'NoneType'
            return True
        case _:
            return False


def parse_file(path: str) -> Module:
    with open(path) as f:
        tree = ast.parse(f.read())
    module = module_to_type(tree, Path(path).stem)
    return module


TYPESHED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../typeshed_mini'))
MODULES = Module('typeshed',
                 typed_dict([make_row(index, file.split('.')[0], parse_file(f'{TYPESHED_DIR}/{file}'))
                             for index, file in enumerate(os.listdir(TYPESHED_DIR))]))


def main() -> None:
    pretty_print_type(MODULES)


def is_bound_method(t: TypeExpr) -> bool:
    return (
        isinstance(t, FunctionType) and t.side_effect.bound_method or
        isinstance(t, Overloaded) and any(is_bound_method(x) for x in t.items)
    )


def get_side_effect(applied: Overloaded) -> SideEffect:
    return SideEffect(
        new=any(x.side_effect.new for x in applied.items),
        update=join_all(x.side_effect.update for x in applied.items),
        bound_method=any(is_bound_method(x) for x in applied.items),
        points_to_args=any(x.side_effect.points_to_args for x in applied.items),
    )
