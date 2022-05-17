from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TypeVar, Protocol, Type, Generic, TypeAlias, Final
import graph_utils as gu

import tac

T = TypeVar('T')


# mix of domain and analysis-specific choice of operations
# nothing here really works...
class Lattice(Protocol):
    def copy(self: T) -> T:
        ...

    @classmethod
    def initial(cls: Type[T]) -> T:
        ...

    @classmethod
    def top(cls: Type[T]) -> T:
        ...

    @classmethod
    def bottom(cls: Type[T]) -> T:
        ...

    def join(self: T, other) -> T:
        ...

    def meet(self: T, other) -> T:
        ...

    def is_bottom(self) -> bool:
        ...

    def is_top(self) -> bool:
        ...


@dataclass(frozen=True)
class Top:
    def __str__(self):
        return '⊤'


@dataclass(frozen=True)
class Bottom:
    def __str__(self):
        return '⊥'


BOTTOM = Bottom()
TOP = Top()


class Evaluator(Protocol[T]):
    def call(self, function: T, args: list[T]) -> T:
        ...

    def binary(self, left: T, right: T) -> T:
        ...

    def const(self, value: object) -> T:
        ...

    def name(self, name: str) -> T:
        ...


def evaluate(evaluate: Evaluator, expr: tac.Expr, values: defaultdict[str, T]) -> T:
    match expr:
        case tac.Call: return evaluate.call(
            function=values[expr.function.name],
            args=[values[arg.name] for arg in expr.args]
        )
        case tac.Binary: return evaluate.binary(
            left=values[expr.left.name],
            right=values[expr.right.name]
        )
        case tac.Const: return evaluate.const(expr.value)
    return TOP


class Cartesian(Generic[T]):
    evaluate: Evaluator[T]
    values: defaultdict[str, T] | Bottom

    def __init__(self, transformer: Type[T], values: dict[str, T] | Bottom):
        super().__init__()
        self.transformer = transformer
        if isinstance(values, Bottom):
            self.values = BOTTOM
        else:
            self.values = defaultdict(lambda: TOP)
            self.values.update(values)

    @staticmethod
    def name() -> str:
        return "Cartesian"

    def __le__(self, other):
        return self.join(other).values == other.values

    def __eq__(self, other):
        return self.values == other.values

    def copy(self: T) -> T:
        return Cartesian(self.transformer, self.values.copy())

    @property
    def is_bottom(self) -> bool:
        return isinstance(self.values, Bottom)

    def join(self: T, other: T) -> T:
        if self.is_bottom:
            return other.copy()
        if other.is_bottom:
            return self.copy()
        res = self.top()
        for k in self.values.keys() | other.values.keys():
            if k in self.values.keys() and k in other.values.keys():
                res.values[k] = self.values[k].join(other.values[k])
            else:
                res.values[k] = TOP
        res.normalize()
        return res

    def transfer(self, ins: tac.Tac, location: str) -> None:
        if self.is_bottom:
            return
        values = self.values.copy()
        for var in tac.gens(ins):
            if var in self.values:
                del self.values[var.name]
        if isinstance(ins, tac.For):
            ins = ins.as_call()
        if isinstance(ins, tac.Assign):
            self.values[ins.lhs.name] = evaluate(self.evaluate, ins.expr, values)
        self.normalize()

    def normalize(self) -> None:
        for k, v in list(self.values.items()):
            if isinstance(v, Bottom):
                self.values = BOTTOM
                return
            if isinstance(v, Top):
                del self.values[k]

    def __str__(self) -> str:
        if self.is_bottom:
            return f'Vars({BOTTOM})'
        return 'Vars({})'.format(", ".join(f'{k}: {v.value}' for k, v in self.values.items()))

    def __repr__(self) -> str:
        return self.values.__repr__()

    def keep_only_live_vars(self, alive_vars: set[str]) -> None:
        for var in set(self.values.keys()) - alive_vars:
            del self.values[var]


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
