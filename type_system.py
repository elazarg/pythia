from __future__ import annotations

import enum
import os
import typing
from dataclasses import dataclass
from pathlib import Path
import ast

from frozendict import frozendict


class TypeExpr:
    pass


T = typing.TypeVar('T', bound=TypeExpr)
K = typing.TypeVar('K')


@dataclass(frozen=True)
class Literal(TypeExpr, typing.Generic[K]):
    value: K

    def __repr__(self):
        return f'Literal[{self.value!r}]'


@dataclass(frozen=True)
class Ref(TypeExpr):
    name: str

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class TypeVar(TypeExpr):
    name: str
    is_args: bool = False

    def __repr__(self):
        return f'*{self.name}' if self.is_args else self.name


@dataclass(frozen=True)
class Index:
    number: typing.Optional[Literal[int]]
    name: typing.Optional[Literal[str]]

    def __repr__(self):
        if self.number is None:
            if self.name is None:
                return 'None'
            return f'{self.name.value}'
        if self.name is None:
            return f'{self.number.value}'
        return f'({self.number.value}){self.name.value}'

    def __sub__(self, other: int) -> Index:
        if self.number is None:
            return self
        if isinstance(other, Index):
            other = other.number.value
        return Index(Literal(self.number.value - other), self.name)

    def __le__(self, other: Index) -> bool:
        if self.number is None or other.number is None:
            return True
        return self.number.value <= other.number.value

    def __lt__(self, other: Index) -> bool:
        if self.number is None or other.number is None:
            return False
        return self.number.value < other.number.value


@dataclass(frozen=True)
class Row(TypeExpr):
    # an index has either a name or an index or both
    index: Index
    type: TypeExpr

    def __repr__(self):
        return f'{self.index}: {self.type}'

    def __le__(self, other: Row) -> bool:
        return self.index <= other.index

def make_row(number: int, name: str, type: TypeExpr) -> Row:
    if number is not None:
        number = Literal(number)
    if name is not None:
        name = Literal(name)
    return Row(Index(number, name), type)


R = typing.TypeVar('R', bound=Row)

@dataclass(frozen=True)
class Union(TypeExpr, typing.Generic[T]):
    items: frozenset[T]

    def __repr__(self) -> str:
        return f'{{{" | ".join(f"{decl}" for decl in self.items)}}}'


@dataclass(frozen=True)
class Intersection(TypeExpr, typing.Generic[R]):
    items: frozenset[R]

    def __repr__(self) -> str:
        items = self.items
        if all(isinstance(item, Row) for item in items):
            res = ", ".join(f"{item.index}={item.type}"
                            for item in sorted(items, key=lambda item: item.index))
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

    def __and__(self, other: TypeExpr) -> Intersection[R]:
        if not isinstance(other, Intersection):
            raise TypeError(f'Cannot intersect {self} with {other}')
        if not self.items:
            return other
        if not other.items:
            return self
        res = Intersection(self.items | other.items)
        assert len(set(type(item) for item in res.items)) <= 1
        return res

    def split_by_index(self: Intersection[Row], index: Literal[int] | Literal[str]) -> tuple[Intersection[Row], Intersection[Row]]:
        index = index_from_literal(index)
        good = intersect([item for item in self.items if match_index(item.index, index)])
        bad = intersect([item for item in self.items if not match_index(item.index, index)])
        return good, bad

    def squeeze(self: Intersection[Row]) -> Intersection[T] | T:
        if len(self.items) == 1:
            return next(iter(self.items))
        return self


TypedDict: typing.TypeAlias = Intersection[Row]


@dataclass(frozen=True)
class Class:
    name: str
    class_dict: TypedDict
    inherits: tuple[TypeExpr, ...]
    protocol: bool
    type_params: tuple[TypeVar, ...]

    def __repr__(self):
        return f'{self.name}'


@dataclass(frozen=True)
class Module:
    name: str
    class_dict: TypedDict

    def __repr__(self):
        return f'module {self.name}({self.class_dict})'


@dataclass(frozen=True)
class FunctionType(TypeExpr):
    params: TypedDict
    return_type: TypeExpr
    new: Literal[bool]
    property: Literal[bool]
    type_params: tuple[TypeVar, ...]

    def __repr__(self) -> str:
        if self.property == Literal(True):
            return f'->{self.return_type}'
        else:
            type_params = ', '.join(str(x) for x in self.type_params)
            return f'[{type_params}]({self.params} -> {"new " if self.new.value else ""}{self.return_type})'

    def is_property(self) -> bool:
        return self.params is None


@dataclass(frozen=True)
class Instantiation(TypeExpr):
    generic: TypeExpr
    type_args: tuple[TypeExpr]

    def __repr__(self):
        return f'{self.generic}[{", ".join(repr(x) for x in self.type_args)}]'


def constant(value: object) -> TypeExpr:
    if value is None:
        t = Ref('builtins.NoneType')
    else:
        t = Ref(f'builtins.{type(value).__name__}')
    return intersect([t, Literal(value)])


def simplify_generic(t, context: dict[TypeVar, TypeExpr]):
    match t:
        case Module(name, class_dict):
            return t
        case Ref():
            return t
        case TypeVar():
            return context.get(t, t)
        case Literal():
            return t
        case Intersection(items):
            return intersect(simplify_generic(item, context) for item in items)
        case Union(items):
            return union(simplify_generic(item, context) for item in items)
        case Row(index, typeexpr):
            return Row(index, simplify_generic(typeexpr, context))
        case FunctionType(params, return_type, new, property, type_params):
            new_params = simplify_generic(params, context)
            new_return_type = simplify_generic(return_type, context)
            return FunctionType(new_params, new_return_type, new, property, type_params)
        case Class(name, class_dict, inherits, protocol, type_params):
            class_dict = simplify_generic(class_dict, context)
            inherits = tuple(simplify_generic(x, context) for x in inherits)
            return Class(name, class_dict, inherits, protocol, type_params)
        case Instantiation(generic, type_args):
            match generic, type_args:
                case (Class(type_params=type_params) | FunctionType(type_params=type_params)), type_args:
                    unpacked_args = unpack_type_args(type_args, context)
                    star_index = None
                    for k, value in enumerate(generic.type_params):
                        if generic.type_params[k].is_args:
                            star_index = k
                            break
                    if star_index is None:
                        assert len(type_params) == len(unpacked_args)
                        new_context = context.copy()
                        for k, v in zip(type_params, unpacked_args):
                            new_context[k] = v
                        return simplify_generic(generic, new_context)
                    left_params, star_param, right_params = type_params[:star_index], type_params[star_index], type_params[star_index + 1:]
                    left_args, star_args = unpacked_args[:len(left_params)], unpacked_args[len(left_params):]
                    star_args, right_args = star_args[:len(star_args)-len(right_params)], star_args[-len(right_params):]
                    new_context = context.copy()
                    new_context[star_param] = intersect(make_row(i, None, arg) for i, arg in enumerate(star_args))
                    for k, v in zip(left_params + right_args, left_args + right_args):
                        new_context[k] = v
                    return simplify_generic(generic, new_context)
                case generic, type_args if all(isinstance(ta, TypeVar) for ta in type_args):
                    new_type_args = []
                    for ta in type_args:
                        if ta.is_args:
                            args = context.get(ta, None)
                            if args is None:
                                new_type_args.append(ta)
                            else:
                                assert isinstance(args, Intersection)
                                for arg in sorted(args.items, key=lambda x: x.index):
                                    assert isinstance(arg, Row)
                                    new_type_args.append(arg.type)
                        else:
                            new_type_args.append(context.get(ta, ta))
                    new_type_args = tuple(new_type_args)
                    if new_type_args == type_args:
                        return Instantiation(simplify_generic(generic, context), type_args)
                    return simplify_generic(Instantiation(generic, new_type_args), context)
                case Ref() as ref, type_args:
                    # avoid infinite recursion when instantiating a generic class with its own parameter
                    return t
                    g = resolve_static_ref(ref)
                    assert isinstance(g, (Class, FunctionType)), f'{g!r} is not a generic in {t!r}'
                    t = Instantiation(g, type_args)
                    return simplify_generic(t, context)
                case Row(expected_index, t), (Literal() as actual_index_literal,):
                    if match_index(expected_index, index_from_literal(actual_index_literal)):
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


def unpack_type_args(type_args: typing.Iterable[TypeExpr], context: dict[TypeVar, TypeExpr]) -> tuple[TypeExpr, ...]:
    unpacked_args = []
    for arg in type_args:
        simplified_arg = simplify_generic(arg, context)
        if isinstance(arg, TypeVar) and arg.is_args:
            assert isinstance(simplified_arg, Intersection)
            items = simplified_arg.items
            assert all(isinstance(x, Row) for x in items)
            assert all(x.index is not None for x in items)
            unpacked_args.extend([arg.type for arg in sorted(items, key=lambda x: x.index)])
        else:
            unpacked_args.append(simplified_arg)
    return tuple(unpacked_args)


def intersect(rows: typing.Iterable[R]) -> TypedDict:
    return Intersection[R](frozenset(rows))


def union(items: typing.Iterable[TypeExpr]) -> Union:
    return Union(frozenset(items))


TOP = intersect([])
BOTTOM = union([])



def join(t1: TypeExpr, t2: TypeExpr):
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
            return union([t1, t2])
        case (Intersection(items), other) | (other, Intersection(items)):
            return Intersection(items & {other}).squeeze()
        case (Ref() as ref, other) | (other, Ref() as ref):
            # this = resolve_static_ref(ref)
            # assert isinstance(this, Class), f'{this!r} is not a class'
            return union([ref, other])
        case Literal(value1), Literal(value2):
            if value1 == value2:
                return Literal(value1)
            else:
                return union([t1, t2])
        case FunctionType(params1, return_type1, new1), FunctionType(params2, return_type2, new2):
            return FunctionType(meet(params1,  params2), join(return_type1, return_type2), join(new1, new2))
        case Class(), Class():
            return TOP
        case Class(), _:
            return TOP
        case _: raise NotImplementedError(f'{t1!r}, {t2!r}')


def join_all(items: typing.Iterable[TypeExpr]) -> TypeExpr:
    res = BOTTOM
    for t in items:
        res = join(res, t)
    return res


def meet(t1: TypeExpr, t2: TypeExpr):
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
        case (Union(items1), Union(items2)):
            raise NotImplementedError
        case (Union(items), t) | (t, Union(items)):
            if t in items:
                return t
            return join_all(meet(item, t) for item in items)
        case (Intersection(items1), Intersection(items2)):
            return Intersection(items1 | items2)
        case (Intersection(items), Row() as r) | (Row() as r, Intersection(items)):
            return Intersection(items | {r})
        case (Intersection(items), t) | (t, Intersection(items)):
            return Intersection(items | {t})
        case (FunctionType() as f1, FunctionType() as f2):
            return intersect([f1, f2])
        case (t1, t2):
            if t1 == t2:
                return t1
            else:
                return intersect([t1, t2])

def meet_all(items: typing.Iterable[TypeExpr]) -> TypeExpr:
    res = TOP
    for t in items:
        res = meet(res, t)
    return res


class Action(enum.Enum):
    INDEX = enum.auto()
    SELECT = enum.auto()


def match_row(param: Row, arg: Row) -> bool:
    return match_index(param.index, arg.index)


def index_from_literal(index: Literal) -> Index:
    return Index(index, None) if isinstance(index.value, int) else Index(None, index)


def match_index(param: Index, arg: Index) -> bool:
    return arg.number == param.number and arg.number is not None or arg.name == param.name and arg.name is not None


def unify_argument(type_params: tuple[TypeVar, ...], param: TypeExpr, arg: TypeExpr) -> typing.Optional[dict[TypeVar, TypeExpr]]:
    match param, arg:
        case TypeVar(), arg:
            return {param: arg}
        case Ref() as param, Ref() as arg:
            if param == arg:
                return {}
            return None
        case Ref() as param, arg:
            if resolve_static_ref(param) == arg:
                return {}
            return unify_argument(type_params, resolve_static_ref(param), arg)
        case param, Ref() as arg:
            return unify_argument(type_params, param, resolve_static_ref(arg))
        case Class() as param, Instantiation(Class(), args):
            res = simplify_generic(arg, {})
            return res
        case param, Instantiation(Ref() as param_type, param_args) if not isinstance(param, Ref):
            return unify_argument(type_params, param, Instantiation(resolve_static_ref(param_type), param_args))
        case Instantiation(param, param_args), Instantiation(arg, arg_args):
            if param != arg:
                return None
            collect = [unify_argument(type_params, param, arg) for param, arg in zip(param_args, arg_args)]
            collect = [x for x in collect if x is not None]
            result = {k: join_all(item[k] for item in collect) for k in type_params}
            return result
        case Literal() as param, Literal() as arg:
            if param == arg:
                 return {}
            return None
        case Row() as param, Row() as arg:
            if match_row(param, arg):
                return unify_argument(type_params, param.type, arg.type)
            return None
        case Class(), Literal():
            return None
        case param, Intersection(items):
            res = [unify_argument(type_params, param, item) for item in items]
            res = [x for x in res if x is not None]
            return {k: join_all(item[k] for item in res) for k in type_params}
        case param, arg:
            return None
            raise NotImplementedError(f'{param!r}, {arg!r}')


def unify(type_params: tuple[TypeVar, ...], params: Intersection[Row], args: Intersection[Row]) -> typing.Optional[dict[TypeVar, TypeExpr]]:
    if not params.items:
        return {}
    bound_types: dict[TypeVar, TypeExpr] = {}
    assert isinstance(args, Intersection), f'{args!r}'
    unbound_args = set(args.items)
    varparam = None
    for param in params.items:
        if isinstance(param.type, TypeVar) and param.type.is_args:
            varparam = param
            continue
        matching_args = {arg for arg in args.items if match_row(param, arg)}
        unbound_args -= matching_args
        bindings = [unify_argument(type_params, param, arg) for arg in matching_args]
        if None in bindings:
            return None
        for k in type_params:
            bound_types[k] = join(bound_types.get(k, BOTTOM), join_all(bind.get(k, BOTTOM) for bind in bindings))
    if varparam is not None:
        if not unbound_args:
            argtype = TOP
        else:
            minimal_index = min(arg.index for arg in unbound_args)
            argtype = intersect(
                Row(r.index - minimal_index, r.type)
                for r in unbound_args)
        bound_types[varparam.type] = argtype
    return bound_types


def apply(t: TypeExpr, action: Action, arg: TypeExpr) -> TypeExpr:
    t = simplify_generic(t, {})
    arg = simplify_generic(arg, {})
    match t, action, arg:
        case t, action, Intersection(items):
            mid = [apply(t, action, item) for item in items]
            res = meet_all(mid)
            return res
        case Intersection(items), action, arg:
            mid = [apply(item, action, arg) for item in items]
            res = join_all(mid)
            return res
        case Union(items), action, arg:
            mid = [apply(item, action, arg) for item in items]
            return join_all(mid)
        case t, action, Union(items):
            mid = [apply(t, action, item) for item in items]
            return join_all(mid)
        case Ref() as ref, action, arg:
            return apply(resolve_static_ref(ref), action, arg)
        case FunctionType(Intersection(items), return_type=return_type, new=new, property=property, type_params=type_params), Action.SELECT, Row() as arg:
            res = set()
            for param in items:
                if match_row(param, arg):
                    res.add(unify((), param, arg))
            return FunctionType(intersect(res), return_type=return_type, new=new, property=property, type_params=type_params)
        # case Generic(type_params=type_params, type=FunctionType() as f), Action.SELECT, arg:
        #     # Curry the function
        #     this, other = f.params.split_by_index(Literal(arg.index))
        #     bound_types = unify(type_params, this, intersect([arg]))
        #     return_type = simplify_generic(f.return_type, bound_types)
        #     params = intersect(Row(row.index, simplify_generic(row.type, bound_types))
        #                        for row in other.items)
        #     type_params = tuple(t for t in type_params if t not in bound_types)
        #     res = FunctionType(params, return_type=return_type, new=f.new, property=f.property)
        #     if type_params:
        #         res = Generic(type_params, res)
        #     return res
        case Class(class_dict=class_dict), Action.SELECT, arg:
            init = apply(class_dict, Action.INDEX, Literal[str]('__call__'))
            return apply(init, Action.SELECT, arg)
        case Row() as t, Action.INDEX, Literal() as arg_index:
            if match_index(t.index, index_from_literal(arg_index)):
                return t.type
            return BOTTOM
        case Row(index, t), Action.INDEX, Ref(name):
            if name == 'builtins.int' and index.number is not None:
                return t
            return BOTTOM
        case Instantiation(generic=FunctionType()|Class()) | TypeVar() as t, action, arg:
            new_t = simplify_generic(t, {})
            if new_t != t:
                return apply(new_t, action, arg)
            assert False, f'Cannot apply {action} to {t} with {arg}'
        case Instantiation(generic=Ref() as generic, type_args=args) as t, action, arg:
            new_generic = resolve_static_ref(generic)
            if new_generic != generic:
                return apply(Instantiation(new_generic, args), action, arg)
            assert False, f'Cannot apply {action} to {t} with {arg}'
        case Literal(value), action, arg:
            return TOP
        case Module() as t, Action.INDEX, arg:
            return apply(t.class_dict, Action.INDEX, arg)
        case Class(class_dict=class_dict) | Module(class_dict=class_dict), Action.INDEX, Literal(str() as index):
            good, bad = class_dict.split_by_index(arg)
            if not good.items:
                raise TypeError(f'No attribute {index!r} in {t}')
            types = [x.type for x in good.items]
            if isinstance(t, Module):
                return meet_all(types)
            else:
                bound_types = [bind_self(t, x) for x in types]
                for i in range(len(bound_types)):
                    if isinstance(bound_types[i], FunctionType):
                        if bound_types[i].property.value:
                            bound_types[i] = bound_types[i].return_type
                return meet_all(bound_types)
        case Class(), Action.INDEX, arg:
            getter = apply(t, Action.INDEX, Literal[str]('__getitem__'))
            res = call(getter, intersect([make_row(0, None, arg)]))
            return res
        case _:
            raise NotImplementedError(f'{t=}, {action=}, {arg=}')


def bind_self(t: Class, f: FunctionType) -> FunctionType:
    if not isinstance(f, FunctionType):
        return f
    curried_params = intersect(Row(r.index-1, r.type) for r in f.params.items if r.index.number != Literal(0))
    res = FunctionType(curried_params, return_type=f.return_type, new=f.new, property=f.property, type_params=f.type_params[1:])
    if f.type_params:
        res = simplify_generic(res, {f.type_params[0]: t})
    return res


def subscr(t: TypeExpr, index: TypeExpr | Module | Class) -> TypeExpr:
    return apply(t, Action.INDEX, index)


def call(t: TypeExpr, args: Intersection[Row]) -> TypeExpr:
    res = t
    match t:
        case Intersection(items):
            ts = [call(item, args) for item in items]
            res = meet_all(ts)
            return res
        case FunctionType(params, return_type, new, property, type_params):
            binding = unify(tuple(type_params), params, args)
            if binding is None:
                return TOP
            return simplify_generic(return_type, binding)
    for item in args.items:
        res = apply(res, Action.SELECT, item)
    if res == BOTTOM:
        return BOTTOM
    assert isinstance(res, FunctionType), f'{t!r}, {args!r}, {res!r}'
    assert res.params == TOP
    return res.return_type


def make_constructor(t: TypeExpr) -> TypeExpr:
    args = TypeVar('Args', is_args=True)
    return_type = intersect([args, Instantiation(t, (args,))])
    return FunctionType(params=intersect([make_row(None, None, args)]),
                        return_type=return_type,
                        new=Literal(True),
                        property=Literal(False),
                        type_params=(args,))


def make_slice_constructor() -> TypeExpr:
    return_type = Ref('builtins.slice')
    return FunctionType(params=intersect([make_row(0, 'start', Ref('builtins.int')),
                                          make_row(1, 'end', Ref('builtins.int'))]),
                        return_type=return_type,
                        new=Literal(True),
                        property=Literal(False),
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
    return subscr(left, Literal[str](lop))


def binop(left: TypeExpr, right: TypeExpr, op: str) -> TypeExpr:
    binop_func = get_binop(left, right, op)
    return call(binop_func, intersect([make_row(0, None, right)]))


def get_unop(left: TypeExpr, op: str) -> TypeExpr:
    return subscr(left, Literal[str](unop_to_dunder_method(op)))


def resolve_static_ref(ref):
    return resolve_relative_ref(ref, MODULES)


def resolve_relative_ref(ref, module):
    result = module
    for attr in ref.name.split('.'):
        result = subscr(result, Literal[str](attr))
    return result


def infer_self(row: Row) -> Row:
    function = row.type
    if not isinstance(function, FunctionType):
        return row
    params = function.params
    if not params:
        raise TypeError(f'Cannot bind self to {function}')
    self_arg, other_args = params.split_by_index(Literal[int](0))
    assert len(self_arg.items) == 1
    self_arg = next(iter(self_arg.items))
    type_params = function.type_params
    if self_arg.type == Ref('typing.Any'):
        self_type = TypeVar('Self')
        self_arg = make_row(0, self_arg.index.name, self_type)
        type_params = (self_type, *type_params)
    params = intersect([self_arg]) & other_args
    g = FunctionType(params, function.return_type, function.new, function.property, type_params)
    return Row(row.index, g)


def pretty_print_type(t: Module | TypeExpr, indent=0):
    match t:
        case Intersection(items):
            for row in items:
                pretty_print_type(row, indent)
        case Row(index, typeexpr):
            print(' ' * indent, index.name.value, '=', end='', sep='')
            if isinstance(typeexpr, Intersection):
                print()
            pretty_print_type(typeexpr, indent)
        case FunctionType(params, return_type, Literal(new), Literal(property), type_params):
            pretty_params = ', '.join(f'{row.index.name.value}: {row.type}' for row in sorted(params.items, key=lambda x: x.index))
            pretty_type_params = ', '.join(str(x) for x in type_params)
            print(f'[{pretty_type_params}]({pretty_params}) -> {"new " if new else ""}{return_type}')
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
            print(f'({number.value}){name.value}')
        case Literal(value):
            print(f'{value}', end='')
        case _:
            raise NotImplementedError(f'{t!r}, {type(t)}')


def module_to_type(module: ast.Module, name: str) -> Module:
    def free_vars(node: ast.expr):
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    def expr_to_type(expr: ast.expr) -> TypeExpr:
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
                    case ref:
                        items = (expr_to_type(ref),)
                return Instantiation(generic, items)
            case ast.Attribute(value=value, attr=attr):
                ref = expr_to_type(value)
                assert isinstance(ref, Ref), f'Expected Ref, got {ref!r} for {value!r}'
                return Ref(f'{ref.name}.{attr}')
            case _:
                raise NotImplementedError(f'{expr!r}')

    def parse_generic_arguments(slice: ast.Tuple | ast.Name | ast.JoinedStr) -> tuple[TypeVar, ...]:
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
            case ast.AnnAssign(target=ast.Name(id=name), annotation=annotation, value=value):
                yield make_row(index, name, expr_to_type(annotation))
            case ast.Import(names=aliases):
                for alias in aliases:
                    name = alias.name
                    asname = alias.asname
                    if asname is None:
                        continue
                    yield make_row(index, asname, Ref(name))
            case ast.ClassDef(name=name, body=body, bases=base_expressions, decorator_list=decorator_list):
                base_classes = []
                protocol = False
                type_params: tuple[TypeVar] = ()
                for base in base_expressions:
                    match base:
                        case ast.Name(id='Protocol'):
                            protocol = True
                        case ast.Subscript(value=ast.Name(id=('Protocol' | 'Generic') as id), slice=params):
                            protocol = id == 'Protocol'
                            type_params = parse_generic_arguments(params)
                        case ast.Name(id=id):
                            base_classes.append(Ref(id))
                base_classes = tuple(base_classes)

                metaclass = Class(f'__{name}_metaclass__',
                                  intersect([
                                      make_row(0, '__call__', FunctionType(intersect([
                                          make_row(0, 'cls', TypeVar('Infer')),
                                      ]), Ref('type'),
                                          new=Literal[bool](True),
                                          property=Literal[bool](False),
                                          type_params=())),
                                  ]),
                                  inherits=(Ref('builtins.type'),),
                                  protocol=False,
                                  type_params=())

                class_dict = intersect([infer_self(row) for index, stmt in enumerate(body)
                                        for row in stmt_to_rows(stmt, index)])
                res = Class(name, class_dict, inherits=base_classes, protocol=protocol,
                            type_params=type_params)

                yield make_row(index, name, res)
            case ast.FunctionDef(name=name, returns=returns, decorator_list=decorator_list,
                                 args=ast.arguments(args=args, vararg=vararg, kwonlyargs=kwonlyargs,
                                                    kw_defaults=kw_defaults, kwarg=kwarg, defaults=defaults)):
                freevars = {x for node in args for x in free_vars(node)}
                params = intersect(
                    [make_row(index, arg.arg, expr_to_type(arg.annotation))
                     for index, arg in enumerate(args)]
                )
                returns = expr_to_type(returns)
                name_decorators = [decorator.id for decorator in decorator_list if isinstance(decorator, ast.Name)]
                new = Literal[bool](returns not in [Ref(f'buitlins.{x}') for x in ['int', 'float', 'bool', 'None']])
                property = Literal[bool]('property' in name_decorators)
                type_params = tuple(TypeVar(x) for x in freevars if x in generic_vars)
                f = FunctionType(params, returns, new=new, property=property, type_params=type_params)

                yield make_row(index, name, f)
            case ast.If():
                return
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


def parse_file(path: str) -> Module:
    with open(path) as f:
        tree = ast.parse(f.read())
    module = module_to_type(tree, Path(path).stem)
    return module


MODULES = Module('typeshed',
                 intersect([make_row(index, file.split('.')[0], parse_file(f'typeshed_mini/{file}'))
                            for index, file in enumerate(os.listdir('typeshed_mini'))]))


def main():
    pretty_print_type(MODULES)


if __name__ == '__main__':
    main()


# SimpleType: TypeAlias = Ref | TypeVar | OverloadedFunctionType | MapType | Generic | Instantiation | FunctionType
#
# def instantiate(self: T, type_args: dict[TypeVar, SimpleType]) -> T: pass
#
# def bind(self: T, type_args: dict[TypeVar, SimpleType]) -> T: pass
#
#
# def make_tuple(t: ObjectType) -> ObjectType:
#     return ObjectType('tuple', frozendict({
#         '__getitem__': make_function_type(t),
#     }))
#
#
# def iter_method(element_type: ObjectType) -> OverloadedFunctionType:
#     return make_function_type(ObjectType(f'iterator', frozendict({
#         '__next__': make_function_type(element_type),
#     }), type_params=frozendict({'Iterator': element_type})), new=True)
#
#
# OBJECT = ObjectType('object')
# FLOAT = ObjectType('float')
# INT = ObjectType('int')
# STRING = ObjectType('str')
# BOOL = ObjectType('bool', frozendict({
#     '__bool__': make_function_type(Ref('bool'), new=False),
# }))
# TYPE = ObjectType('type', frozendict({
#     '__call__': make_function_type(TypeVar('T'), new=False),
# }), type_params=frozendict({'T': None}))
# NONE = ObjectType('None')
# CODE = ObjectType('code')
# ASSERTION_ERROR = ObjectType('AssertionError')
# SLICE = ObjectType('slice')
#
# LIST = ObjectType('list', frozendict({
#     '__getitem__': make_function_type(TypeVar('T'), new=False),
#     '__iter__': iter_method(TypeVar('T')),
#     '__len__': make_function_type(INT, new=False),
#     '__contains__': make_function_type(BOOL, new=False),
#     'clear': make_function_type(NONE, new=False),
#     'copy': make_function_type(NONE, new=False),
#     'count': make_function_type(INT, new=False),
#     'extend': make_function_type(NONE),
#     'index': make_function_type(INT, new=False),
#     'insert': make_function_type(NONE, new=False),
#     'pop': make_function_type(TypeVar('T')),
#     'remove': make_function_type(NONE, new=False),
#     'reverse': make_function_type(NONE, new=False),
#     'sort': make_function_type(NONE, new=False),
# }), type_params=frozendict({'T': None}))
#
# DICT = ObjectType('dict', frozendict({
#     '__getitem__': make_function_type(OBJECT, new=False),
#     '__setitem__': make_function_type(NONE, new=False),
# }))
#
# TUPLE_CONSTRUCTOR = make_function_type(make_tuple(OBJECT), new=False)
# LIST_CONSTRUCTOR = make_function_type(LIST, ParamsType(varargs=True), new=False)
# SLICE_CONSTRUCTOR = make_function_type(SLICE, new=False)
# DICT_CONSTRUCTOR = make_function_type(DICT, new=True)
#
# NDARRAY = ObjectType('ndarray', frozendict({
#     'mean': make_function_type(FLOAT, new=False, pure=True),
#     'std': make_function_type(FLOAT, new=False, pure=True),
#     'shape': make_tuple(INT),
#     'size': INT,
#     '__getitem__': OverloadedFunctionType(
#         tuple([
#             FunctionType(ParamsType((INT,), varargs=False), FLOAT, new=True),
#             FunctionType(ParamsType((FLOAT,), varargs=False), FLOAT, new=True),  # Fix: ad-hoc
#             FunctionType(ParamsType((SLICE,), varargs=False), Ref('numpy.ndarray'), new=True),
#             FunctionType(ParamsType((Ref('numpy.ndarray'),), varargs=False), Ref('numpy.ndarray'), new=True),
#             FunctionType(ParamsType((make_tuple(OBJECT),), varargs=False), OBJECT, new=True),  # FIX: not necessarily new
#         ])
#     ),
#     '__iter__': iter_method(FLOAT),  # inaccurate
#     'T': Property(Ref('numpy.ndarray'), new=True),
#     'astype': make_function_type(Ref('numpy.ndarray'), new=True, pure=True),
#     'reshape': make_function_type(Ref('numpy.ndarray'), new=True, pure=False),
#     'ndim': INT,
# }))
#
# ARRAY_GEN = make_function_type(NDARRAY, new=True)
#
# DATAFRAME = ObjectType('DataFrame', frozendict({
#     'loc': DICT,
# }))
#
# TIME_MODULE = ObjectType('/time')

# SKLEARN_MODULE = ObjectType('/sklearn', frozendict({
#     'metrics': ObjectType('/metrics', frozendict({
#         'log_loss': make_function_type(FLOAT, new=False),
#         'accuracy_score': make_function_type(FLOAT, new=False),
#         'f1_score': make_function_type(FLOAT, new=False),
#         'precision_score': make_function_type(FLOAT, new=False),
#         'recall_score': make_function_type(FLOAT, new=False),
#         'roc_auc_score': make_function_type(FLOAT, new=False),
#         'average_precision_score': make_function_type(FLOAT, new=False),
#         'roc_curve': make_function_type(make_tuple(FLOAT), new=False),
#         'confusion_matrix': make_function_type(make_tuple(INT), new=False),
#     })),
#     'linear_model': ObjectType('/linear_model', frozendict({
#         'LogisticRegression': make_function_type(ObjectType('LogisticRegression', frozendict({
#             'fit': make_function_type(ObjectType('Model', frozendict({
#                 'predict': make_function_type(FLOAT),
#                 'predict_proba': ARRAY_GEN,
#                 'score': make_function_type(FLOAT),
#             })), new=False),
#         })), new=True),
#         'LinearRegression': make_function_type(ObjectType('LinearRegression', frozendict({
#             'fit': make_function_type(ObjectType('Model', frozendict({
#                 'predict': make_function_type(FLOAT),
#                 'predict_proba': ARRAY_GEN,
#                 'score': make_function_type(FLOAT),
#             })), new=False),
#         })), new=True),
#     })),
# }))
#
# PANDAS_MODULE = ObjectType('/pandas', frozendict({
#     'DataFrame': TYPE[DATAFRAME],
# }))
#
# BUILTINS_MODULE = ObjectType('/builtins', frozendict({
#     'range': make_function_type(ObjectType('range', frozendict({
#         '__iter__': iter_method(INT),
#     }))),
#     'zip': make_function_type(ObjectType('zip', frozendict({
#         '__iter__': iter_method(make_tuple(OBJECT)),
#     }))),
#
#     'len': make_function_type(INT, new=False),
#     'print': make_function_type(NONE, new=False),
#     'abs': make_function_type(FLOAT, new=False),
#     'round': make_function_type(FLOAT, new=False),
#     'min': make_function_type(FLOAT, new=False),
#     'max': make_function_type(FLOAT, new=False),
#     'sum': make_function_type(FLOAT, new=False),
#     'all': make_function_type(BOOL, new=False),
#     'any': make_function_type(BOOL, new=False),
#
#     'int': TYPE[INT],
#     'float': TYPE[FLOAT],
#     'str': TYPE[STRING],
#     'bool': TYPE[BOOL],
#     'code': TYPE[CODE],
#     'object': TYPE[OBJECT],
#     'AssertionError': TYPE[ASSERTION_ERROR],
# }))
#
# FUTURE_MODULE = ObjectType('/future', frozendict({
#     'annotations': ObjectType('_'),
# }))
#
# MATPLOTLIB_MODULE = ObjectType('/matplotlib', frozendict({
#     'pyplot': ObjectType('pyplot', frozendict({
#         'plot': make_function_type(NONE, new=False),
#         'show': make_function_type(NONE, new=False),
#     })),
# }))
#
# PERSIST_MODULE = ObjectType('/persist', frozendict({
#     'range': make_function_type(ObjectType('range', frozendict({
#         '__iter__': iter_method(INT),
#     }))),
# }))
#
# MODULES = frozendict({
#     '__future__': FUTURE_MODULE,
#     'numpy': NUMPY_MODULE,
#     'pandas': PANDAS_MODULE,
#     'time': TIME_MODULE,
#     'sklearn': SKLEARN_MODULE,
#     'matplotlib': MATPLOTLIB_MODULE,
#     'persist': PERSIST_MODULE,
#     'builtins': BUILTINS_MODULE,
# })
#
#
# def resolve_module(path: str, modules: dict = MODULES) -> ObjectType:
#     path = path.split('.')
#     obj = modules[path[0]]
#     for part in path[1:]:
#         obj = obj.fields[part]
#     return obj
#
#
# def transpose(double_dispatch_table: dict[tuple[SimpleType, SimpleType], dict[str, tuple[SimpleType, bool]]]
#               ) -> dict[str, OverloadedFunctionType]:
#     ops = ['+', '-', '*', '/', '**', '//', '%', '@',
#            '==', '!=', '<', '<=', '>', '>=']
#     result: dict[str, list[FunctionType]] = {op: [] for op in ops}
#     for op in ops:
#         for params, table in double_dispatch_table.items():
#             if op in table:
#                 return_type, new = table[op]
#                 result[op].append(FunctionType(ParamsType(params), return_type, new))
#     return {op: OverloadedFunctionType(tuple(result[op])) for op in ops}
#
#
# BINARY = transpose({
#     (INT, INT): {
#         '/':   (FLOAT, False),
#         **{op: (INT, False) for op in ['+', '-', '*',       '**', '//', '%']},
#         **{op: (INT, False) for op in ['&', '|', '^', '<<', '>>']},
#         **{op: (BOOL, False) for op in ['<', '>', '<=', '>=', '==', '!=']},
#     },
#     (INT, FLOAT): {
#         **{op: (FLOAT, False) for op in ['+', '-', '*', '/', '**', '//', '%']},
#         **{op: (BOOL, False) for op in ['<', '>', '<=', '>=', '==', '!=']},
#     },
#     (FLOAT, INT): {
#         **{op: (FLOAT, False) for op in ['+', '-', '*', '/', '**', '//', '%']},
#         **{op: (BOOL, False) for op in ['<', '>', '<=', '>=', '==', '!=']},
#     },
#     (FLOAT, FLOAT): {
#         **{op: (FLOAT, False) for op in ['+', '-', '*', '/', '**', '//', '%']},
#         **{op: (BOOL, False) for op in ['<', '>', '<=', '>=', '==', '!=']},
#     },
#     (NDARRAY, NDARRAY): {op: (NDARRAY, True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
#     (NDARRAY, INT):     {op: (NDARRAY, True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
#     (INT, NDARRAY):     {op: (NDARRAY, True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
#     (NDARRAY, FLOAT):   {op: (NDARRAY, True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
#     (FLOAT, NDARRAY):   {op: (NDARRAY, True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
# })