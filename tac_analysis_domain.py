from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Protocol, Type, Generic, TypeAlias
import graph_utils as gu

T = TypeVar('T')


# mix of domain and analysis-specific choice of operations
# nothing here really works...
class AbstractDomain(Protocol):
    @staticmethod
    def name() -> str:
        raise NotImplementedError

    def copy(self: T) -> T:
        ...

    @classmethod
    def view(cls, cfg: gu.Cfg[T]) -> IterationStrategy[T]:
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

    def transfer(self, ins: T) -> None:
        ...

    def keep_only_live_vars(self, vars: set):
        pass


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
