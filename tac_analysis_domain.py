from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TypeVar, Protocol, Type, Generic, TypeAlias, Final, Callable, Iterator, Iterable
import graph_utils as gu

import tac

T = TypeVar('T')


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

    def call(self, function: T, args: list[T]) -> T:
        return self.top()

    def binary(self, left: T, right: T, op: str) -> T:
        return self.top()

    def predefined(self, name: tac.Predefined) -> T:
        return self.top()

    def const(self, value: object) -> T:
        return self.top()

    def attribute(self, var: T, attr: str) -> T:
        return self.top()

    def subscr(self, array: T, index: T) -> T:
        return self.top()

    def imported(self, modname: str) -> T:
        return self.top()

    def annotation(self, string: str) -> T:
        return self.top()

    def name(self) -> str:
        return self.top()

    def var(self, value: T) -> T:
        return value


@dataclass(frozen=True)
class Top:
    def __str__(self):
        return '⊤'

    def copy(self: T) -> T:
        return self


@dataclass(frozen=True)
class Bottom:
    def __str__(self):
        return '⊥'

    def copy(self: T) -> T:
        return self


BOTTOM = Bottom()
TOP = Top()


class Map(Generic[T]):
    map: defaultdict[tac.Var, T]

    def __init__(self, dict: dict[tac.Var, T] = None):
        self.map = defaultdict(lambda: TOP)
        if dict is not None:
            self.update(dict)

    def __getitem__(self, key: tac.Var) -> T:
        assert isinstance(key, tac.Var)
        return self.map[key]

    def __setitem__(self, key: tac.Var, value: T):
        assert isinstance(key, tac.Var)
        if isinstance(value, Top):
            if key in self.map:
                del self.map[key]
        else:
            self.map[key] = value

    def __iter__(self):
        return iter(self.map)

    def __contains__(self, key: tac.Var):
        return key in self.map

    def __len__(self):
        return len(self.map)

    def __eq__(self, other: Map[T] | Bottom) -> bool:
        return isinstance(other, Map) and self.map == other.map

    def __delitem__(self, key: tac.Var):
        del self.map[key]

    def __repr__(self):
        items = ', '.join(f'{k}={v}' for k, v in self.items())
        return f'Map({items})'

    def __str__(self):
        return repr(self)

    def items(self) -> list[tuple[tac.Var, T]]:
        return list(self.map.items())

    def keys(self) -> set[tac.Var]:
        return set(self.map.keys())

    def copy(self):
        return Map(self.map)

    def update(self, dictionary: dict[tac.Var, T]):
        for k, v in dictionary.items():
            self[k] = v


MapDomain: TypeAlias = Map[T] | Bottom


def normalize(values: MapDomain[T]) -> MapDomain:
    if isinstance(values, Bottom):
        return BOTTOM
    result = values.copy()
    for k, v in values.items():
        if isinstance(v, Bottom):
            return BOTTOM
        if isinstance(v, Top):
            del result[k]
    return result


class Cartesian(Generic[T]):
    lattice: Lattice[T]

    # TODO: this does not belong here
    def view(self, cfg: gu.Cfg[T]):
        return ForwardIterationStrategy(cfg)

    def __init__(self, lattice: Lattice[T]):
        super().__init__()
        self.lattice = lattice

    def name(self) -> str:
        return f"Cartesian({self.lattice.name()})"

    def is_less_than(self, left: T, right: T) -> bool:
        return self.join(left, right) == right

    def is_equivalent(self, left, right) -> bool:
        return self.is_less_than(left, right) and self.is_less_than(right, left)

    def copy(self: T) -> T:
        return Cartesian(self.lattice)

    def is_bottom(self, values) -> bool:
        return isinstance(values, Bottom)

    def initial(self, annotations: dict[tac.Var, str]) -> MapDomain[T]:
        result = Map()
        result.update({
            name: self.lattice.annotation(t)
            for name, t in annotations.items()
        })
        return result

    def top(self) -> MapDomain[T]:
        return self.initial({})

    def bottom(self) -> MapDomain[T]:
        return BOTTOM

    def join(self, left: MapDomain[T], right: MapDomain[T]) -> MapDomain[T]:
        if self.is_bottom(left):
            return right.copy()
        if self.is_bottom(right):
            return left.copy()
        res = self.top()
        for k in left.keys() | right.keys():
            if k in left.keys() and k in right.keys():
                res[k] = self.lattice.join(left[k], right[k])
        return normalize(res)

    @staticmethod
    def transformer_expr(lattice: Lattice[T], values: Map[T], expr: tac.Expr) -> T:
        def eval(expr: tac.Expr | tac.Predefined) -> T:
            return Cartesian.transformer_expr(lattice, values, expr)

        match expr:
            case tac.Call():
                return lattice.call(
                    function=eval(expr.function),
                    args=[eval(arg) for arg in expr.args]
                )
            case tac.Binary():
                return lattice.binary(
                    left=eval(expr.left),
                    right=eval(expr.right),
                    op=expr.op
                )
            case tac.Predefined:
                expr: tac.Predefined = expr
                return lattice.predefined(expr)
            case tac.Const():
                return lattice.const(expr.value)
            case tac.Attribute():
                return lattice.attribute(eval(expr.var), expr.attr.name)
            case tac.Subscript():
                return lattice.subscr(eval(expr.var), eval(expr.index))
            case tac.Yield():
                return TOP
            case tac.Import():
                return lattice.imported(expr.modname)
            case tac.Var():
                return lattice.var(values[expr])
            case _:
                assert False, f'unexpected expr {expr}'

    @staticmethod
    def transformer_signature(lattice: Lattice[T], value: T, signature: tac.Signature) -> Map[T]:
        match signature:
            case tuple():
                value_tuple = lattice.assign_tuple(value)
                return Map({signature[i]: value_tuple[i] for i in range(len(value_tuple))})
            case tac.Var():
                return Map({signature: lattice.assign_var(value)})
            case _:
                assert False, f'unexpected signature {signature}'

    def transfer(self, values: MapDomain[T], ins: tac.Tac, location: str) -> MapDomain[T]:
        if isinstance(values, Bottom):
            return BOTTOM
        transformer = Cartesian.transformer_expr(self.lattice, values.copy())
        if isinstance(ins, tac.For):
            ins = ins.as_call()
        if isinstance(ins, tac.Assign):
            if isinstance(ins.lhs, (tac.Var, tac.Predefined)):
                values[ins.lhs] = transformer(ins.expr)
        return normalize(values)

    @staticmethod
    def back_transformer(lattice: Lattice[T], assigned: T, expr: tac.Expr | tac.Predefined) -> Map[T]:
        match expr:
            case tac.Call():
                return lattice.back_call(assigned)
            case tac.Binary():
                return lattice.back_binary(assigned)
            case tac.Predefined():
                return lattice.back_predefined(assigned)
            case tac.Const():
                return lattice.back_const(assigned)
            case tac.Attribute():
                return lattice.back_attribute(assigned)
            case tac.Subscript():
                return lattice.back_subscr(assigned)
            case tac.Yield():
                return lattice.back_yield(assigned)
            case tac.Import():
                return lattice.back_imported(expr.modname)
            case tac.Var():
                return lattice.back_var(assigned)
            case _:
                assert False, f'unexpected expr {expr}'

    @staticmethod
    def back_transformer_signature(lattice: Lattice[T], values: Map[T], signature: tac.Signature) -> T:
        match signature:
            case tuple():
                value = tuple(values[v] for v in signature)
                return lattice.back_assign_tuple(value)
            case tac.Var():
                return lattice.back_assign_var(values[signature])
            case _:
                assert False, f'unexpected signature {signature}'

    def back_transfer(self, values: MapDomain[T], ins: tac.Tac, location: str) -> MapDomain[T]:
        if isinstance(values, Bottom):
            return BOTTOM
        transformer = Cartesian.back_transformer(self.lattice, values.copy())
        for var in tac.free_vars(ins) & values.keys():
            del values[var]
        if isinstance(ins, tac.For):
            ins = ins.as_call()
        if isinstance(ins, tac.Assign):
            if isinstance(ins.lhs, (tac.Var, tac.Predefined)):
                values[ins.lhs] = transformer(ins.expr)
        return normalize(values)

    def keep_only_live_vars(self, values: MapDomain[T], alive_vars: set[tac.Var]) -> None:
        for var in set(values.keys()) - alive_vars:
            del values[var]


class AbstractAnalysis(Protocol[T]):
    def name(self) -> str:
        raise NotImplementedError

    def view(self, cfg: gu.Cfg[T]) -> IterationStrategy[T]:
        raise NotImplementedError

    def transfer(self, inv: Cartesian, ins: T, location: str) -> None:
        raise NotImplementedError

    def initial(self) -> T:
        raise NotImplementedError

    def top(self) -> T:
        raise NotImplementedError

    def bottom(self) -> T:
        raise NotImplementedError


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

GLOBALS: Final[Object] = Object('globals()')
