import abc
from dataclasses import dataclass
import typing

from spyte import graph_utils as gu
from spyte.graph_utils import Label


class IterationStrategy[T]:
    @property
    @abc.abstractmethod
    def entry_label(self) -> Label: ...
    @abc.abstractmethod
    def successors(self, label: Label) -> typing.Iterator[Label]: ...
    @abc.abstractmethod
    def predecessors(self, label: Label) -> typing.Iterator[Label]: ...
    @abc.abstractmethod
    def __getitem__(self, label: Label) -> gu.Block[T]: ...
    @abc.abstractmethod
    def order[K, Q](self, pair: tuple[K, Q]) -> tuple[K, Q]: ...


@dataclass
class ForwardIterationStrategy[T](IterationStrategy[T]):
    cfg: gu.Cfg[T]

    @property
    def entry_label(self) -> Label:
        return self.cfg.entry_label

    def successors(self, label: Label) -> typing.Iterator[Label]:
        return self.cfg.successors(label)

    def predecessors(self, label: Label) -> typing.Iterator[Label]:
        return self.cfg.predecessors(label)

    def __getitem__(self, label: Label) -> gu.Block[T]:
        return self.cfg[label]

    def order[K, Q](self, pair: tuple[K, Q]) -> tuple[K, Q]:
        return pair


@dataclass
class BackwardIterationStrategy[T](IterationStrategy[T]):
    cfg: gu.Cfg[T]

    @property
    def entry_label(self) -> Label:
        return self.cfg.exit_label

    def successors(self, label: Label) -> typing.Iterator[Label]:
        return self.cfg.predecessors(label)

    def predecessors(self, label: Label) -> typing.Iterator[Label]:
        return self.cfg.successors(label)

    def __getitem__(self, label: Label) -> gu.BackwardBlock[T]:
        return typing.cast(gu.BackwardBlock, reversed(self.cfg[label]))

    def order[K, Q](self, pair: tuple[K, Q]) -> tuple[Q, K]:
        return pair[1], pair[0]


def iteration_strategy[T](cfg: gu.Cfg[T], backward: bool) -> IterationStrategy[T]:
    return BackwardIterationStrategy(cfg) if backward else ForwardIterationStrategy(cfg)
