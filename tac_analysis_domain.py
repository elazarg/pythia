from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TypeVar, Protocol, Type, Generic, TypeAlias, Final
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


def transfer(lattice: Lattice[T], expr: tac.Expr | tac.Predefined, values: defaultdict[tac.Name, T]) -> T:
    match expr:
        case tac.Call(): return lattice.call(
            function=values[expr.function],
            args=[values[arg] for arg in expr.args]
        )
        case tac.Binary(): return lattice.binary(
            left=values[expr.left],
            right=values[expr.right],
            op=expr.op
        )
        case tac.Predefined(): return lattice.predefined(expr)
        case tac.Const(): return lattice.const(expr.value)
        case tac.Attribute(): return lattice.attribute(values[expr.var], expr.attr.name)
        case tac.Subscript(): return lattice.subscr(values[expr.var], values[expr.index])
        case tac.Yield(): return TOP
        case tac.Import(): return lattice.imported(expr.modname)
        case tac.Var(): return values[expr]
        case str(): assert False, f'unexpected string {expr}'
        case _: assert False, f'unexpected expr {expr}'


class Cartesian(Generic[T]):
    lattice: Lattice[T]
    values: defaultdict[tac.Name, T] | Bottom

    # TODO: this does not belong here
    def view(self, cfg: gu.Cfg[T]):
        return ForwardIterationStrategy(cfg)

    def __init__(self, lattice: Lattice[T], values: dict[tac.Name, T] | Bottom = BOTTOM):
        super().__init__()
        self.lattice = lattice
        if isinstance(values, Bottom):
            self.values = BOTTOM
        else:
            self.values = defaultdict(lambda: TOP)
            for c in tac.Predefined:
                if (value := lattice.predefined(c)) is not None:
                    self.values[c] = value
            self.values.update(values)

    def set_initial(self, annotations: dict[str, str]):
        if self.is_bottom:
            self.__init__(self.lattice, {})
        else:
            self.values.clear()
        self.values.update({
            name: self.lattice.annotation(t)
            for name, t in annotations.items()
        })

    @staticmethod
    def name() -> str:
        return "Cartesian"

    def __le__(self, other):
        return self.join(other).values == other.values

    def __eq__(self, other):
        return self.values == other.values

    def copy(self: T) -> T:
        return Cartesian(self.lattice, self.values.copy())

    @property
    def is_bottom(self) -> bool:
        return isinstance(self.values, Bottom)

    def top(self) -> T:
        return Cartesian(self.lattice, {})

    def bottom(self) -> T:
        return Cartesian(self.lattice, BOTTOM)

    def join(self: T, other: T) -> T:
        if self.is_bottom:
            return other.copy()
        if other.is_bottom:
            return self.copy()
        res = self.top()
        for k in self.values.keys() | other.values.keys():
            if k in self.values.keys() and k in other.values.keys():
                res.values[k] = self.join(self.values[k], other.values[k])
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
                del self.values[var]
        if isinstance(ins, tac.For):
            ins = ins.as_call()
        if isinstance(ins, tac.Assign):
            if isinstance(ins.lhs, (tac.Var, tac.Predefined)):
                self.values[ins.lhs] = transfer(self.lattice, ins.expr, values)
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
        return 'Vars({})'.format(", ".join(f'{k}: {v}' for k, v in self.values.items()))

    def __repr__(self) -> str:
        return self.values.__repr__()

    def keep_only_live_vars(self, alive_vars: set[tac.Var]) -> None:
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
