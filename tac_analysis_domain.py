from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, Protocol, Generic, TypeAlias, Final, Iterator, Type, Callable, Iterable
import graph_utils as gu

import tac

T = TypeVar('T')
K = TypeVar('K')


@dataclass
class ForwardIterationStrategy(Generic[T]):
    cfg: gu.Cfg[T]

    @property
    def entry_label(self):
        return self.cfg.entry_label

    def successors(self, label):
        return self.cfg.successors(label)

    def __getitem__(self, label) -> gu.Block:
        return self.cfg[label]


@dataclass
class BackwardIterationStrategy(Generic[T]):
    cfg: gu.Cfg[T]

    @property
    def entry_label(self):
        return self.cfg.exit_label

    def successors(self, label):
        return self.cfg.predecessors(label)

    def __getitem__(self, label) -> gu.Block:
        return gu.BackwardBlock(self.cfg[label])


IterationStrategy: TypeAlias = ForwardIterationStrategy | BackwardIterationStrategy


class Lattice(Generic[T]):
    def copy(self: T) -> T:
        ...

    def top(self) -> T:
        ...

    def bottom(self) -> T:
        ...

    def join(self, left: T, right: T) -> T:
        ...

    def meet(self, left: T, right: T) -> T:
        ...

    def is_bottom(self, elem: T) -> bool:
        ...

    def is_top(self, elem: T) -> bool:
        ...

    def const(self, value: object) -> T:
        return self.top()

    def var(self, value: T) -> T:
        return value

    def attribute(self, var: T, attr: tac.Var) -> T:
        assert isinstance(attr, tac.Var)
        return self.top()

    def subscr(self, array: T, index: T) -> T:
        return self.top()

    def call(self, function: T, args: list[T]) -> T:
        return self.top()

    def binary(self, left: T, right: T, op: str) -> T:
        return self.top()

    def unary(self, value: T, op: tac.UnOp) -> T:
        return tac.AllocationType.NONE

    def predefined(self, name: tac.Predefined) -> T:
        return self.top()

    def imported(self, modname: str) -> T:
        return self.top()

    def annotation(self, string: str) -> T:
        return self.top()

    def assign_tuple(self, values: T) -> list[T]:
        return self.top()

    def assign_var(self, value: T) -> T:
        return value

    def name(self) -> str:
        raise NotImplementedError

    def back_call(self, assigned, size: int) -> tuple[T, list[T]]:
        return (self.top(), [])

    def back_binary(self, value: T) -> tuple[T, T]:
        return (self.top(), self.top())

    def back_unary(self, value: T) -> T:
        return self.top()

    def back_predefined(self, value: T) -> None:
        return None

    def back_const(self, value: T) -> None:
        return None

    def back_attribute(self, value: T) -> T:
        return self.top()

    def back_subscr(self, value: T) -> tuple[T, T]:
        return (self.top(), self.top())

    def back_imported(self, value: T) -> None:
        return

    def back_annotation(self, value: T) -> T:
        return self.top()

    def back_var(self, value: T) -> T:
        return value

    def back_yield(self, value: T) -> T:
        return self.top()

    def back_assign_var(self, value: T) -> T:
        return value

    def back_assign_tuple(self, values: tuple[T]) -> T:
        return self.top()

    def back_assign_subscr(self, var: T, index: T) -> T:
        return self.top()

    def back_assign_attribute(self, var: T, attr: tac.Var) -> T:
        return self.top()

    def back_return(self) -> T:
        return self.top()

    def default(self) -> T:
        return self.top()

    def back_inplace_binary(self, lhs: T, rhs: T, op: str) -> tuple[T, T]:
        raise NotImplementedError

    # TODO: Fix defaults, make it a proper lattice

    def allocation_type_function(self, function: T) -> tac.AllocationType:
        return tac.AllocationType.NONE

    def allocation_type_binary(self, left: T, right: T, op: str) -> tac.AllocationType:
        return tac.AllocationType.NONE

    def allocation_type_unary(self, value: T, op: tac.UnOp) -> tac.AllocationType:
        return tac.AllocationType.NONE

    def allocation_type_attribute(self, val, name) -> tac.AllocationType:
        return tac.AllocationType.NONE


@dataclass(frozen=True)
class Top:
    def __str__(self):
        return '⊤'

    def copy(self: T) -> T:
        return self

    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self

    def __and__(self, other):
        return other

    def __rand__(self, other):
        return other


@dataclass(frozen=True)
class Bottom:
    def __str__(self):
        return '⊥'

    def copy(self: T) -> T:
        return self

    def __or__(self, other):
        return other

    def __ror__(self, other):
        return other

    def __rand__(self, other):
        return self

    def __and__(self, other):
        return self


BOTTOM = Bottom()
TOP = Top()


class Map(Generic[K, T]):
    # Essentially a defaultdict, but a defaultdict make values appear out of nowhere
    _map: dict[tac.Var, T]
    default: T

    def __init__(self, default: T, d: dict[tac.Var, T] = None):
        self.default = default
        self._map = {}
        if d is not None:
            self.update(d)

    def __getitem__(self, key: tac.Var) -> T:
        if not isinstance(key, tac.Var):
            breakpoint()
        assert isinstance(key, tac.Var), key
        return self._map.get(key, self.default)

    def __setitem__(self, key: tac.Var, value: T):
        assert isinstance(key, tac.Var), key
        if value == self.default:
            if key in self._map:
                del self._map[key]
        else:
            assert not isinstance(value, dict)
            assert not isinstance(value, Map)
            self._map[key] = value

    def update(self, dictionary: dict[tac.Var, T] | Map) -> None:
        for k, v in dictionary.items():
            self[k] = v

    def __iter__(self):
        return iter(self._map)

    def __contains__(self, key: tac.Var):
        return key in self._map

    def __len__(self):
        return len(self._map)

    def __eq__(self, other: Map[T] | Bottom) -> bool:
        return isinstance(other, Map) and self._map == other._map

    def __delitem__(self, key: tac.Var) -> None:
        del self._map[key]

    def __repr__(self) -> str:
        items = ', '.join(f'{k}={v}' for k, v in self._map.items())
        return f'Map({items})'

    def __str__(self) -> str:
        return repr(self)

    def items(self) -> Iterable[tuple[tac.Var, T]]:
        return self._map.items()

    def values(self) -> Iterable[T]:
        return self._map.values()

    def keys(self) -> set[tac.Var]:
        return self._map.keys()

    def copy(self) -> Map:
        return Map(self.default, self._map)

    def join(self, other, resolution: Callable[[T, T], T] = None) -> Map:
        if resolution is None:
            resolution = self.default.join
        new_map = self.copy()
        for k in self.keys() | other.keys():
            new_map[k] = resolution(new_map[k], other[k])
        return new_map


MapDomain: TypeAlias = Map[K, T] | Bottom


def normalize(values: MapDomain[K, T]) -> MapDomain[K, T]:
    if isinstance(values, Bottom):
        return BOTTOM
    if any(isinstance(v, Bottom) for v in values.values()):
        return BOTTOM
    return values


class Analysis(Protocol[T]):
    backward: bool

    def name(self) -> str:
        ...

    def is_less_than(self, left: T, right: T) -> bool:
        ...

    def is_equivalent(self, left, right) -> bool:
        ...

    def copy(self, values: T) -> T:
        ...

    def is_bottom(self, values) -> bool:
        ...

    def initial(self, annotations: dict[K, str]) -> T:
        ...

    def top(self) -> T:
        ...

    def bottom(self) -> T:
        ...

    def join(self, left: T, right: T) -> T:
        ...

    def transfer(self, values: T, ins: tac.Tac, location: str) -> T:
        ...


class VarAnalysis(Analysis[MapDomain[K, T]]):
    lattice: Lattice[T]
    backward: bool

    def __init__(self, lattice: Lattice[T], backward: bool = False):
        super().__init__()
        self.lattice = lattice
        self.backward = backward

    def name(self) -> str:
        return f"{self.lattice.name()}"

    def is_less_than(self, left: T, right: T) -> bool:
        return self.join(left, right) == right

    def is_equivalent(self, left, right) -> bool:
        return self.is_less_than(left, right) and self.is_less_than(right, left)

    def copy(self, values: MapDomain[K, T]) -> MapDomain[K, T]:
        return values.copy()

    def is_bottom(self, values: MapDomain[K, T]) -> bool:
        return isinstance(values, Bottom)

    def make_map(self, d: dict[K, T] = None) -> MapDomain[K, T]:
        d = d or {}
        return Map(default=self.lattice.default(), d=d)

    def initial(self, annotations: dict[K, str]) -> MapDomain[K, T]:
        result = self.make_map()
        result.update({
            name: self.lattice.annotation(t)
            for name, t in annotations.items()
        })
        return result

    def top(self) -> MapDomain[K, T]:
        return self.make_map()

    def bottom(self) -> MapDomain[K, T]:
        return BOTTOM

    def join(self, left: MapDomain[K, T], right: MapDomain[K, T]) -> MapDomain[K, T]:
        match left, right:
            case (Bottom(), _): return right
            case (_, Bottom()): return left
            case (Map(), Map()):
                res = self.top()
                for k in left.keys() | right.keys():
                    if k in left.keys() and k in right.keys():
                        res[k] = self.lattice.join(left[k], right[k])
                return normalize(res)

    def transformer_expr(self, values: Map[T], expr: tac.Expr) -> T:
        def eval(expr: tac.Expr | tac.Predefined) -> T:
            return self.transformer_expr(values, expr)

        match expr:
            case tac.Var():
                return self.lattice.var(values[expr])
            case tac.Attribute():
                val = eval(expr.var)
                expr.allocation = self.lattice.allocation_type_attribute(val, expr.field)
                return self.lattice.attribute(val, expr.field)
            case tac.Call():
                func = eval(expr.function)
                expr.allocation = self.lattice.allocation_type_function(func)
                return self.lattice.call(
                    function=func,
                    args=[eval(arg) for arg in expr.args]
                )
            case tac.Unary():
                value = eval(expr.var)
                expr.allocation = self.lattice.allocation_type_unary(value, expr.op)
                return self.lattice.unary(value=value, op=expr.op)
            case tac.Binary():
                left = eval(expr.left)
                right = eval(expr.right)
                expr.allocation = self.lattice.allocation_type_binary(left, right, expr.op)
                return self.lattice.binary(left=left, right=right, op=expr.op)
            case tac.Predefined():
                expr: tac.Predefined = expr
                return self.lattice.predefined(expr)
            case tac.Const():
                return self.lattice.const(expr.value)
            case tac.Subscript():
                return self.lattice.subscr(eval(expr.var), eval(expr.index))
            case tac.Yield():
                return self.lattice.top()
            case tac.Import():
                print(f"{expr}: {type(expr)}; {expr.modname} {type(expr.modname)}")
                if isinstance(expr.modname, tac.Attribute):
                    val = eval(expr.modname.var)
                    return self.lattice.attribute(val, expr.modname.field)
                else:
                    return self.lattice.imported(expr.modname)
            case tac.MakeFunction():
                return self.lattice.top()
            case _:
                assert False, f'unexpected expr of type {type(expr)}: {expr}'

    def transformer_signature(self, value: T, signature: tac.Signature) -> Map[T]:
        match signature:
            case tuple():
                value_tuple = self.lattice.assign_tuple(value)
                if not isinstance(value_tuple, tuple):
                    return self.make_map({signature[i]: self.lattice.top() for i in range(len(signature))})
                return self.make_map({signature[i]: value_tuple[i] for i in range(len(value_tuple))})
            case tac.Var():
                return self.make_map({signature: self.lattice.assign_var(value)})
            case tac.Attribute():
                return self.make_map()
            case tac.Subscript():
                return self.make_map()
            case _:
                assert False, f'unexpected signature {signature}'

    def forward_transfer(self, values: MapDomain[K, T], ins: tac.Tac, location: str) -> MapDomain[K, T]:
        if isinstance(values, Bottom):
            return BOTTOM
        if isinstance(ins, tac.For):
            ins = ins.as_call()
        match ins:
            case tac.Assign():
                assigned = self.transformer_expr(values, ins.expr)
                return self.transformer_signature(assigned, ins.lhs)
            case tac.Return():
                return self.make_map({
                    tac.Var('return'): self.transformer_expr(values, ins.value)
                })
            case tac.Del():
                return self.make_map()
        return self.make_map()

    def back_transformer(self, assigned: T, expr: tac.Expr | tac.Predefined) -> Map[T]:
        match expr:
            case tac.Attribute():
                value = self.lattice.back_attribute(assigned)
                return self.make_map({expr.var: value} if isinstance(expr.var, tac.Var) else {})
            case tac.Var():
                return self.make_map({expr: self.lattice.back_var(assigned)})
            case tac.Call():
                (f, args) = self.lattice.back_call(assigned, len(expr.args))
                d = {expr.args[i]: args[i] for i in range(len(expr.args))
                     if isinstance(expr.args[i], tac.Var)}
                if isinstance(expr.function, tac.Var):
                    d[expr.function] = f
                return self.make_map(d)
            case tac.Binary():
                left, right = self.lattice.back_binary(assigned)
                return self.make_map({
                    expr.left: left,
                    expr.right: right
                })
            case tac.Unary():
                return self.make_map({
                    expr.var: self.lattice.back_unary(assigned)
                })
            case tac.Predefined():
                return self.make_map()
            case tac.Const():
                return self.make_map()
            case tac.Subscript():
                array, index = self.lattice.back_subscr(assigned)
                return self.make_map({expr.var: array, expr.index: index})
            case tac.Yield():
                value = self.lattice.back_yield(assigned)
                return self.make_map({expr.value: value})
            case tac.Import():
                return self.make_map()
            case tac.MakeFunction():
                return self.make_map()
            case tac.MakeClass():
                return self.make_map()
            case _:
                assert False, f'unexpected expr {expr}'

    def back_transformer_signature(self, values: Map[T], signature: tac.Signature) -> T:
        match signature:
            case tuple():
                value = tuple(values[v] for v in signature)
                return self.lattice.back_assign_tuple(value)
            case tac.Var():
                return self.lattice.back_assign_var(values[signature])
            case tac.Attribute():
                return self.lattice.back_assign_attribute(self.transformer_expr(values, signature.var), signature.field)
            case tac.Subscript():
                return self.lattice.back_assign_subscr(self.transformer_expr(values, signature.var), self.transformer_expr(values, signature.var))
            case _:
                assert False, f'unexpected signature {signature}'

    def back_transfer(self, values: MapDomain[K, T], ins: tac.Tac, location: str) -> MapDomain[K, T]:
        if isinstance(values, Bottom):
            return BOTTOM
        if isinstance(ins, tac.For):
            ins = ins.as_call()
        if isinstance(ins, tac.Assign):
            assigned = self.back_transformer_signature(values, ins.lhs)
            return self.back_transformer(assigned, ins.expr)
        if isinstance(ins, tac.Return):
            return self.make_map({ins.value: self.lattice.back_return()})
        return values

    def transfer(self, values: MapDomain[K, T], ins: tac.Tac, location: str) -> MapDomain[K, T]:
        if isinstance(values, Bottom):
            return BOTTOM
        values = values.copy()
        if self.backward:
            to_update = self.back_transfer(values, ins, location)
        else:
            to_update = self.forward_transfer(values, ins, location)
        for var in tac.gens(ins):
            if var in values:
                del values[var]
        values.update(to_update)
        return normalize(values)


@dataclass(frozen=True)
class Object:
    location: str

    def __str__(self):
        return f'@{self.location}'

    def __repr__(self):
        return f'@{self.location}'

    def pretty(self, field: tac.Var) -> str:
        if self == LOCALS:
            if field.is_stackvar:
                return '$' + field.name
            return field.name
        return f'{self}.{field.name}'


LOCALS: Final[Object] = Object('locals()')
NONLOCALS: Final[Object] = Object('NONLOCALS')

GLOBALS: Final[Object] = Object('globals()')
