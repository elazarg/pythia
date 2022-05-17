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


# mix of domain and analysis-specific choice of operations
# nothing here really works...
class Lattice(Protocol[T]):
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
        ...

    def binary(self, left: T, right: T, op: str) -> T:
        ...

    def predefined(self, name: tac.Predefined) -> T:
        ...

    def const(self, value: object) -> T:
        ...

    def attribute(self, var: T, attr: str) -> T:
        ...

    def subscr(self, array: T, index: T) -> T:
        ...

    def imported(self, modname: str) -> T:
        ...

    def annotation(self, string: str) -> T:
        ...

    def name(self) -> str:
        ...

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

    def __init__(self):
        self.map = defaultdict(lambda: TOP)

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
        res = Map()
        res.update(self.map.copy())
        return res

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
    def transformer(lattice: Lattice[T], values: Map[T]) -> Callable[[tac.Expr], T]:
        def eval(expr: tac.Expr | tac.Predefined) -> T:
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
                case tac.Predefined():
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
                    return values[expr]
                case str():
                    assert False, f'unexpected string {expr}'
                case _:
                    assert False, f'unexpected expr {expr}'

        return eval

    def transfer(self, values: MapDomain[T], ins: tac.Tac, location: str) -> MapDomain[T]:
        if isinstance(values, Bottom):
            return BOTTOM
        transformer = Cartesian.transformer(self.lattice, values.copy())
        for var in tac.gens(ins) & values.keys():
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
