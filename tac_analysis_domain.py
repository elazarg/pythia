from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Protocol, Type, Generic, TypeAlias, Final
import graph_utils as gu

import tac

T = TypeVar('T')


# mix of domain and analysis-specific choice of operations
# nothing here really works...
class Lattice(Protocol):
    @staticmethod
    def name() -> str:
        raise NotImplementedError

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


class AbstractDomain(Lattice):
    @classmethod
    def view(cls, cfg: gu.Cfg[T]) -> IterationStrategy[T]:
        ...

    def transfer(self, ins: T, location: str) -> None:
        ...

    def keep_only_live_vars(self, vars: set):
        pass


@dataclass(frozen=True)
class Top:
    def __str__(self):
        return '⊤'


@dataclass(frozen=True)
class Bottom:
    def __str__(self):
        return '⊥'


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
