from __future__ import annotations as _

import ast
import enum
import os
import typing
from dataclasses import dataclass, replace
from pathlib import Path


class TypeExpr:
    pass


T = typing.TypeVar('T', bound=TypeExpr)
K = typing.TypeVar('K')


@dataclass(frozen=True)
class Literal(TypeExpr):
    value: int | str | bool | float | None

    def __repr__(self) -> str:
        return f'Literal({self.value!r})'


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
class Row(TypeExpr):
    # an index has either a name or an index or both
    index: Index
    type: TypeExpr

    def __repr__(self) -> str:
        return f'{self.index}: {self.type}'

    def __le__(self, other: Row) -> bool:
        return self.index <= other.index


def make_row(number: typing.Optional[int], name: typing.Optional[str], t: TypeExpr) -> Row:
    return Row(Index(number if number is not None else None,
                     name if name is not None else None), t)


@dataclass(frozen=True)
class Union(TypeExpr):
    items: frozenset[TypeExpr]

    def __repr__(self) -> str:
        return f'{{{" | ".join(f"{item}" for item in self.items)}}}'

    def squeeze(self) -> TypeExpr:
        if len(self.items) == 1:
            return next(iter(self.items))
        return self


@dataclass(frozen=True)
class Intersection(TypeExpr):
    items: frozenset[TypeExpr]

    def __repr__(self) -> str:
        items = self.items
        if all(isinstance(item, Row) for item in items):
            res = ", ".join(f"{item.index}={item.type}"
                            for item in sorted(self.row_items(), key=lambda item: item.index))
            if len(items) == 1:
                return f'({res},)'
            return f'({res})'
        if any(isinstance(item, Literal) and isinstance(item.value, int) for item in items):
            items -= {Ref('builtins.int')}
        if any(isinstance(item, Literal) and isinstance(item.value, str) for item in items):
            items -= {Ref('builtins.str')}
        if any(isinstance(item, Literal) and item.value is None for item in items):
            items -= {Ref('builtins.NoneType')}
        return f'{{{" & ".join(f"{decl}" for decl in items)}}}'

    def __and__(self, other: TypeExpr) -> Intersection:
        if not isinstance(other, Intersection):
            raise TypeError(f'Cannot intersect {self} with {other}')
        if not self.items:
            return other
        if not other.items:
            return self
        res = Intersection(self.items | other.items)
        assert len(set(type(item) for item in res.items)) <= 1
        return res

    def row_items(self) -> frozenset[Row]:
        assert all(isinstance(item, Row) for item in self.items)
        return typing.cast(frozenset[Row], self.items)

    def split_by_index(self: Intersection, value: int | str) -> tuple[Intersection, Intersection]:
        index = index_from_literal(value)
        items = self.row_items()
        good = intersect([item for item in items if match_index(item.index, index)])
        bad = intersect([item for item in items if not match_index(item.index, index)])
        return good, bad

    def squeeze(self: Intersection) -> TypeExpr:
        if len(self.items) == 1:
            return next(iter(self.items))
        return self


TypedDict: typing.TypeAlias = Intersection


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
    instructions: tuple[str, ...]

    def __repr__(self) -> str:
        return f'{self.new}'


@dataclass(frozen=True)
class FunctionType(TypeExpr):
    params: TypedDict
    return_type: TypeExpr
    side_effect: SideEffect
    property: bool
    type_params: tuple[TypeVar, ...]

    def __repr__(self) -> str:
        if self.property == Literal(True):
            return f'->{self.return_type}'
        else:
            type_params = ', '.join(str(x) for x in self.type_params)
            return f'[{type_params}]({self.params} -> {"new " if self.new() else ""}{self.return_type})'

    def is_property(self) -> bool:
        return self.params is None

    def new(self):
        return self.side_effect.new


@dataclass(frozen=True)
class Instantiation(TypeExpr):
    generic: TypeExpr
    type_args: tuple[TypeExpr, ...]

    def __repr__(self) -> str:
        return f'{self.generic}[{", ".join(repr(x) for x in self.type_args)}]'


def constant(value: object) -> TypeExpr:
    assert isinstance(value, (int, str, type(None), bool, float)), value
    if value is None:
        t = Ref('builtins.NoneType')
    else:
        t = Ref(f'builtins.{type(value).__name__}')
    return intersect([t, Literal(value)])


def simplify_generic(t: TypeExpr, context: dict[TypeVar, TypeExpr]) -> TypeExpr:
    # TODO: simplify to ClassInstance
    match t:
        case Module():
            return t
        case Ref():
            return t
        case TypeVar() as t:
            return context.get(t, t)
        case Literal():
            return t
        case Intersection(items):
            return intersect(simplify_generic(item, context) for item in items)
        case Union(items):
            return union(simplify_generic(item, context) for item in items)
        case Row() as row:
            return replace(row, type=simplify_generic(row.type, context))
        case FunctionType(params=params, return_type=return_type) as function:
            new_params = simplify_generic(params, context)
            assert isinstance(new_params, Intersection)
            new_return_type = simplify_generic(return_type, context)
            return replace(function, params=new_params, return_type=new_return_type)
        case Class(class_dict=class_dict, inherits=inherits) as klass:
            new_class_dict = simplify_generic(class_dict, context)
            assert isinstance(new_class_dict, Intersection)
            inherits = tuple(simplify_generic(x, context) for x in inherits)
            return replace(klass, class_dict=new_class_dict, inherits=inherits)
        case Instantiation(generic, type_args):
            match generic, type_args:
                case generic, type_args if all(isinstance(ta, TypeVar) for ta in type_args):
                    new_type_args_list: list[TypeExpr] = []
                    for ta in type_args:
                        assert isinstance(ta, TypeVar)
                        if ta.is_args:
                            args = context.get(ta, None)
                            if args is None:
                                new_type_args_list.append(ta)
                            else:
                                assert isinstance(args, Intersection)
                                for arg in sorted(args.row_items(), key=lambda x: x.index):
                                    new_type_args_list.append(arg.type)
                        else:
                            new_type_args_list.append(context.get(ta, ta))
                    new_type_args: tuple[TypeExpr, ...] = tuple(new_type_args_list)
                    if new_type_args == type_args:
                        return Instantiation(simplify_generic(generic, context), type_args)
                    return simplify_generic(Instantiation(generic, new_type_args), context)
                case (Class(type_params=type_params) | FunctionType(type_params=type_params)) as generic, type_args:
                    unpacked_args = unpack_type_args(type_args, context)
                    star_index = None
                    for n, type_param in enumerate(type_params):
                        if type_params[n].is_args:
                            star_index = n
                            break
                    new_context = context.copy()
                    if star_index is not None:
                        left_params, star_param, right_params = type_params[:star_index], type_params[star_index], type_params[star_index + 1:]
                        left_end_index = len(left_params)
                        left_args, star_args = unpacked_args[:left_end_index], unpacked_args[left_end_index:]
                        right_start_index = len(star_args) - len(right_params)
                        star_args, right_args = star_args[:right_start_index], star_args[right_start_index:]
                        new_context[star_param] = intersect(make_row(i, None, arg) for i, arg in enumerate(star_args))
                        type_params = left_params + right_params
                        unpacked_args = left_args + right_args
                    assert len(type_params) == len(unpacked_args), (type_params, unpacked_args)
                    for type_var, v in zip(type_params, unpacked_args):
                        new_context[type_var] = v
                    generic = replace(generic, type_params=())
                    return simplify_generic(generic, new_context)
                case Ref() as ref, type_args:
                    # avoid infinite recursion when instantiating a generic class with its own parameter
                    return t
                    g = resolve_static_ref(ref)
                    assert isinstance(g, (Class, FunctionType)), f'{g!r} is not a generic in {t!r}'
                    t = Instantiation(g, type_args)
                    return simplify_generic(t, context)
                case Row(expected_index, t), (Literal(str() | int() as value),):
                    if match_index(expected_index, index_from_literal(value)):
                        return simplify_generic(t, context)
                    return BOTTOM
                case Row(index, t), (Ref('builtins.int'),):
                    if index.number is not None:
                        return simplify_generic(t, context)
                    return BOTTOM
                case Row() as r, (Intersection(items=items),):
                    x = [simplify_generic(Instantiation(r, (item,)), context) for item in items]
                    return meet_all(x)
                case Intersection(items), type_args:
                    x = [simplify_generic(Instantiation(item, type_args), context) for item in items]
                    return join_all(x)
                case Union(items), type_args:
                    x = [simplify_generic(Instantiation(item, type_args), context) for item in items]
                    return meet_all(x)
                case TypeVar() as generic, (type_arg,):
                    generic = simplify_generic(generic, context)
                    type_arg = simplify_generic(type_arg, context)
                    if isinstance(type_arg, TypeVar):
                        return Instantiation(generic, (type_arg,))
                    return subscr(generic, type_arg)
                case generic, (Union(items),):
                    return union(simplify_generic(Instantiation(generic, (item,)), context)
                                 for item in items)
                case _:
                    raise NotImplementedError(f'Cannot instantiate {generic!r} of type {type(generic)} with {type_args!r}')

        case _:
            raise NotImplementedError(f'{t!r}, {type(t)}')
    raise AssertionError


def unpack_type_args(type_args: typing.Iterable[TypeExpr], context: dict[TypeVar, TypeExpr]) -> tuple[TypeExpr, ...]:
    unpacked_args = []
    for arg in type_args:
        simplified_arg = simplify_generic(arg, context)
        if isinstance(arg, TypeVar) and arg.is_args:
            assert isinstance(simplified_arg, Intersection)
            unpacked_args.extend([arg.type for arg in sorted(simplified_arg.row_items(),
                                                             key=lambda x: x.index)])
        else:
            unpacked_args.append(simplified_arg)
    return tuple(unpacked_args)


def intersect(rows: typing.Iterable[T]) -> TypedDict:
    return Intersection(frozenset(rows))


def union(items: typing.Iterable[TypeExpr]) -> TypeExpr:
    return Union(frozenset(items)).squeeze()


TOP = intersect([])
BOTTOM = union([])


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
        case Intersection(items1), Intersection(items2):
            if not items1: return t2
            if not items2: return t1
            return union([t1, t2])
        case (Intersection(items), other) | (other, Intersection(items)):  # type: ignore
            return Intersection(items & {other}).squeeze()
        case Union(items1), Union(items2):
            return Union(items1 | items2).squeeze()
        case (Union(items), other) | (other, Union(items)):  # type: ignore
            return Union(items | {other}).squeeze()
        case (Ref() as ref, other) | (other, Ref() as ref):  # type: ignore
            return union([ref, other])
        case Literal(value1), Literal(value2):
            if value1 == value2:
                return Literal(value1)
            else:
                return union([t1, t2])
        case FunctionType() as f1, FunctionType() as f2:
            return union([f1, f2])
        case Class(), Class():
            return TOP
        case (Class(name="int") | Ref('builtins.int') as c, Literal(int())) | (Literal(int()), Class(name="int") | Ref('builtins.int') as c):
            return c
        case Class(), _:
            return TOP
        case _: raise NotImplementedError(f'{t1!r}, {t2!r}')


def join_all(items: typing.Iterable[TypeExpr]) -> TypeExpr:
    res: TypeExpr = BOTTOM
    for t in items:
        res = join(res, t)
    return res


def meet(t1: TypeExpr, t2: TypeExpr) -> TypeExpr:
    if t1 == TOP:
        return t2
    if t2 == TOP:
        return t1
    if t1 == BOTTOM or t2 == BOTTOM:
        return BOTTOM
    match (t1, t2):
        case (Literal(value1), Literal(value2)):
            if value1 == value2:
                return Literal(value1)
            else:
                return BOTTOM
        case (Union(), Union()):
            raise NotImplementedError
        case (Union(items), t) | (t, Union(items)):  # type: ignore
            if t in items:
                return t
            return join_all(meet_all([item, t]) for item in items)
        case (Intersection(items1), Intersection(items2)):
            return Intersection(items1 | items2)
        case (Intersection(items), Row() as r) | (Row() as r, Intersection(items)):
            return Intersection(items | {r})
        case (Intersection(items), t) | (t, Intersection(items)):  # type: ignore
            return Intersection(items | {t})
        case (FunctionType() as f1, FunctionType() as f2):
            return intersect([f1, f2])
        case (t1, t2):
            if t1 == t2:
                return t1
            else:
                return intersect([t1, t2])
    raise AssertionError


def meet_all(items: typing.Iterable[TypeExpr]) -> TypeExpr:
    res: TypeExpr = TOP
    for t in items:
        res = meet(res, t)
    if isinstance(res, Intersection):
        return res.squeeze()
    return res


class Action(enum.Enum):
    INDEX = enum.auto()
    SELECT = enum.auto()


def match_row(param: Row, arg: Row) -> bool:
    if not match_index(param.index, arg.index):
        return False
    if param.type == Ref('typing.Any'):
        return True
    return join(param.type, arg.type) == arg.type


def index_from_literal(index: int | str) -> Index:
    return Index(index, None) if isinstance(index, int) else Index(None, index)


def match_index(param: Index, arg: Index) -> bool:
    return arg.number == param.number and arg.number is not None or arg.name == param.name and arg.name is not None


def unify_argument(type_params: tuple[TypeVar, ...], param: TypeExpr, arg: TypeExpr) -> typing.Optional[dict[TypeVar, TypeExpr]]:
    if param == arg:
        return {}
    match param, arg:
        case Ref('typing.Any'), _:
            return {}
        case param, Intersection(items):
            res = [unified for item in items if (unified := unify_argument(type_params, param, item)) is not None]
            if not res:
                return None
            return {k: meet_all(item[k] for item in res) for k in type_params}
        case TypeVar() as param, arg:
            return {param: arg}
        case Class() as param, Instantiation(Class() as arg, arg_args):
            if param.name == arg.name:
                return {k: v for k, v in zip(param.type_params, arg_args)}
            return None
        case param, Instantiation(Ref() as param_type, param_args) if not isinstance(param, Ref):
            return unify_argument(type_params, param, Instantiation(resolve_static_ref(param_type), param_args))
        case Instantiation(param, param_args), Instantiation(arg, arg_args):
            if param != arg:
                return None
            collect = [unified for param, arg in zip(param_args, arg_args)
                       if (unified := unify_argument(type_params, param, arg)) is not None]
            result = {k: join_all(item[k] for item in collect) for k in type_params}
            return result
        case Literal() as param, Literal() as arg:
            if param == arg:
                return {}
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
        case Ref() as param, arg:
            return unify_argument(type_params, resolve_static_ref(param), arg)
        case param, arg:
            return None
            raise NotImplementedError(f'{param!r}, {arg!r}')
    assert False


def unify(type_params: tuple[TypeVar, ...], params: Intersection, args: Intersection) -> typing.Optional[dict[TypeVar, TypeExpr]]:
    if not params.items:
        return {}
    bound_types: dict[TypeVar, TypeExpr] = {}
    assert isinstance(args, Intersection), f'{args!r}'
    unbound_args = set(args.row_items())
    varparam_type: typing.Optional[TypeVar] = None
    for param in params.row_items():
        if isinstance(param.type, TypeVar) and param.type.is_args:
            varparam_type = param.type
            continue
        matching_args = {arg for arg in args.row_items() if match_index(param.index, arg.index)}
        if not matching_args:
            return None
        unbound_args -= matching_args
        bindings = [unified for arg in matching_args
                    if (unified := unify_argument(type_params, param.type, arg.type)) is not None]
        if not bindings:
            return None
        for k in type_params:
            bound_types[k] = join(bound_types.get(k, BOTTOM),
                                  join_all(bind.get(k, BOTTOM) for bind in bindings))
    if varparam_type is not None:
        if not unbound_args:
            argtype = TOP
        else:
            minimal_index = min(arg.index for arg in unbound_args)
            argtype = intersect(
                Row(r.index - minimal_index, r.type)
                for r in unbound_args)
        bound_types[varparam_type] = argtype
    return bound_types


def access(t: TypeExpr, arg: TypeExpr) -> TypeExpr:
    t = simplify_generic(t, {})
    arg = simplify_generic(arg, {})
    match t, arg:
        case Ref() as ref, arg:
            return access(resolve_static_ref(ref), arg)
        case t, Intersection(items):
            mid = [access(t, item) for item in items]
            return meet_all(mid)
        case Union(items), arg:
            mid = [access(item, arg) for item in items]
            return meet_all(mid)
        case Intersection(items), arg:
            mid = [result for item in items if (result := access(item, arg)) != BOTTOM]
            return join_all(mid)
        case t, Union(items):
            mid = [access(t, item) for item in items]
            return join_all(mid)
        case Row() as t, Literal(str() as value):
            if match_index(t.index, index_from_literal(value)):
                if isinstance(t.type, FunctionType) and t.type.property:
                    return t.type.return_type
                return t.type
            return BOTTOM
        case Row() as t, Literal(int() as value):
            if match_index(t.index, index_from_literal(value)):
                return t.type
            return BOTTOM
        case Row(index, t), Ref(name):
            if name == 'builtins.int' and index.number is not None:
                return t
            return BOTTOM
        case Instantiation(generic=FunctionType() | Class() | TypeVar()) as t, arg:
            new_t = simplify_generic(t, {})
            if new_t != t:
                return access(new_t, arg)
            assert False, f'Cannot access {t} with {arg}'
        case Instantiation(generic=Ref() as generic, type_args=args) as t, arg:
            new_generic = resolve_static_ref(generic)
            if new_generic != generic:
                return access(Instantiation(new_generic, args), arg)
            assert False, f'Cannot access {t} with {arg}'
        case Module() as t, arg:
            return access(t.class_dict, arg)
        case Class(class_dict=class_dict) | Module(class_dict=class_dict), Literal(str() as value):
            good, bad = class_dict.split_by_index(value)
            if not good.items:
                return BOTTOM
            types = [x.type for x in good.row_items()]
            if isinstance(t, Module):
                return meet_all(types)
            bound_types = [bind_self(t, x) for x in types
                           if isinstance(x, FunctionType)]
            other_types = [x for x in types if not isinstance(x, FunctionType)]
            accessed_types = [x if not x.property else x.return_type
                              for x in bound_types] + other_types
            return meet_all(accessed_types)
        case Class() as t, arg:
            getter = access(t, Literal('__getitem__'))
            res = call(getter, intersect([make_row(0, None, arg)]))
            return res
        case _:
            raise NotImplementedError(f'{t=}, {arg=}, {type(t)=}, {type(arg)=}')


def bind_self(t: TypeExpr, f: FunctionType) -> FunctionType:
    curried_params = intersect(Row(r.index - 1, r.type) for r in f.params.row_items() if r.index.number != 0)
    res = replace(f, params=curried_params, type_params=f.type_params[1:])
    if f.type_params:
        func = simplify_generic(res, {f.type_params[0]: t})
        assert isinstance(func, FunctionType)
        res = func
    return res


def subscr(t: TypeExpr, index: TypeExpr) -> TypeExpr:
    return access(t, index)


def partial(callable: TypeExpr, args: Intersection) -> TypeExpr:
    match callable:
        case Intersection(items):
            ts = [partial(item, args) for item in items]
            res = join_all(ts)
            return res
        case Union(items):
            ts = [partial(item, args) for item in items]
            res = meet_all(ts)
            return res
        case FunctionType() as f:
            binding = unify(tuple(f.type_params), f.params, args)
            if binding is None:
                return BOTTOM
            return simplify_generic(f, binding)
        case Class(class_dict=class_dict), arg:
            dunder = subscr(class_dict, Literal('__call__'))
            return partial(dunder, arg)
        case _:
            assert False, f'Cannot call {callable} with {args}'


def get_return(callable: TypeExpr) -> TypeExpr:
    match callable:
        case Intersection(items):
            ts = [get_return(item) for item in items]
            res = meet_all(ts)
            return res
        case Union(items):
            ts = [get_return(item) for item in items]
            res = join_all(ts)
            return res
        case FunctionType(return_type=return_type):
            return return_type
        case Class() | Instantiation() | Module() | TypeVar() | Ref() | Literal() | Row():
            assert False, f'{callable!r}'
        case _:
            assert False, f'{callable!r}'
            return BOTTOM


def call(callable: TypeExpr, args: Intersection) -> TypeExpr:
    resolved = partial(callable, args)
    return get_return(resolved)


def make_constructor(t: TypeExpr) -> TypeExpr:
    args = TypeVar('Args', is_args=True)
    return_type = intersect([args, Instantiation(t, (args,))])
    return FunctionType(params=intersect([make_row(None, None, args)]),
                        return_type=return_type,
                        side_effect=SideEffect(new=not is_immutable(return_type), instructions=()),
                        property=False,
                        type_params=(args,))


def make_slice_constructor() -> TypeExpr:
    return_type = Ref('builtins.slice')
    return FunctionType(params=intersect([make_row(0, 'start', Ref('builtins.int')),
                                          make_row(1, 'end', Ref('builtins.int'))]),
                        return_type=return_type,
                        side_effect=SideEffect(new=True, instructions=()),
                        property=False,
                        type_params=())


def binop_to_dunder_method(op: str) -> tuple[str, typing.Optional[str]]:
    match op:
        case '+': return '__add__', '__radd__'
        case '-': return '__sub__', '__rsub__'
        case '*': return '__mul__', '__rmul__'
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
    return subscr(left, Literal(lop))


def binop(left: TypeExpr, right: TypeExpr, op: str) -> TypeExpr:
    binop_func = get_binop(left, right, op)
    if binop_func == BOTTOM:
        # assume there is an implementation.
        return TOP
    return call(binop_func, intersect([make_row(0, None, right)]))


def get_unop(left: TypeExpr, op: str) -> TypeExpr:
    return subscr(left, Literal(unop_to_dunder_method(op)))


def resolve_static_ref(ref: Ref) -> TypeExpr:
    return resolve_relative_ref(ref, MODULES)


def resolve_relative_ref(ref: Ref, module: Module) -> TypeExpr:
    result: TypeExpr = module
    for attr in ref.name.split('.'):
        result = subscr(result, Literal(attr))
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
    if self_arg_row.type == Ref('typing.Any'):
        self_type = TypeVar('Self')
        self_arg_row = replace(self_arg_row, type=self_type)
        type_params = (self_type, *type_params)
    params = intersect([self_arg_row, *other_args.row_items()])
    g = FunctionType(params, function.return_type, function.side_effect, function.property, type_params)
    return Row(row.index, g)


def pretty_print_type(t: Module | TypeExpr, indent: int = 0) -> None:
    match t:
        case Intersection(items):
            for row in items:
                pretty_print_type(row, indent)
        case Row(index, typeexpr):
            if index.name is None:
                print(' ' * indent, end='')
            else:
                print(' ' * indent, index.name, '=', end='', sep='')
            if isinstance(typeexpr, Intersection):
                print()
            pretty_print_type(typeexpr, indent)
        case FunctionType(params, return_type, side_effect, property, type_params):
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
                return Ref(f'typing.Any')
            case ast.Constant(value):
                return constant(value)
            case ast.Name(id=id):
                if id in generic_vars:
                    return generic_vars[id]
                if id in global_aliases:
                    return Ref(global_aliases[id])
                if id in global_names:
                    return Ref(f'{name}.{id}')
                else:
                    return Ref(f'builtins.{id}')
            case ast.Subscript(value=value, slice=slice):
                generic = expr_to_type(value)
                match slice:
                    case ast.Tuple(elts=[arg, ast.Constant(value=x)]) if str(x) == 'Ellipsis':
                        raise NotImplementedError(f'{generic}[{expr_to_type(arg)}, ...]')
                    case ast.Tuple(elts=elts):
                        items = tuple(expr_to_type(x) for x in elts)
                    case ast.Name() as expr:
                        items = (expr_to_type(expr),)
                    case ast.Attribute() as expr:
                        items = (expr_to_type(expr),)
                    case ast.Subscript() as expr:
                        items = (expr_to_type(expr),)
                    case _:
                        raise NotImplementedError(f'{generic}[{expr_to_type(slice)}]')
                return Instantiation(generic, items)
            case ast.Attribute(value=value, attr=attr):
                ref: TypeExpr = expr_to_type(value)
                assert isinstance(ref, Ref), f'Expected Ref, got {ref!r} for {value!r}'
                return Ref(f'{ref.name}.{attr}')
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
                                  intersect([
                                      make_row(0, '__call__', FunctionType(intersect([
                                          make_row(0, 'cls', TypeVar('Infer')),
                                      ]), Ref('type'),
                                          side_effect=SideEffect(new=True, instructions=()),
                                          property=False,
                                          type_params=())),
                                  ]),
                                  inherits=(Ref('builtins.type'),),
                                  protocol=False,
                                  type_params=())

                class_dict = intersect([infer_self(row) for index, stmt in enumerate(body)
                                        for row in stmt_to_rows(stmt, index)])
                res = Class(name, class_dict, inherits=tuple(base_classes_list), protocol=protocol,
                            type_params=type_params)

                yield make_row(index, name, res)
            case ast.FunctionDef() as fdef:
                freevars = {x for node in fdef.args.args for x in free_vars(node)}
                params = intersect(
                    [make_row(index, arg.arg, expr_to_type(arg.annotation))
                     for index, arg in enumerate(fdef.args.args)]
                )
                returns = expr_to_type(fdef.returns)
                name_decorators = [decorator.id for decorator in fdef.decorator_list if isinstance(decorator, ast.Name)]
                side_effect = SideEffect(
                    new=not is_immutable(returns), instructions=()
                )
                property = 'property' in name_decorators
                type_params = tuple(TypeVar(x) for x in freevars if x in generic_vars)
                f = FunctionType(params, returns, side_effect=side_effect, property=property, type_params=type_params)

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
    class_dict = intersect([row for index, stmt in enumerate(module.body)
                            if not isinstance(stmt, ast.Assign)
                            for row in stmt_to_rows(stmt, index)])
    return Module(name, class_dict)


def is_immutable(value: TypeExpr) -> bool:
    match value:
        case Intersection(items):
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


MODULES = Module('typeshed',
                 intersect([make_row(index, file.split('.')[0], parse_file(f'../typeshed_mini/{file}'))
                            for index, file in enumerate(os.listdir('../typeshed_mini'))]))


def main() -> None:
    pretty_print_type(MODULES)


if __name__ == '__main__':
    main()
