# Data flow analysis and stuff.

from __future__ import annotations

import dataclasses
from copy import deepcopy

from networkx.utils import UnionFind
from dataclasses import dataclass
from typing import Type, TypeVar, Optional, ClassVar

import graph_utils
import tac
from tac_analysis_domain import AbstractDomain, IterationStrategy, ForwardIterationStrategy

import graph_utils as gu

T = TypeVar('T')


@dataclass
class Bottom:
    pass


class AliasDomain(AbstractDomain):
    # map stack variable to program variable or constant
    alias: dict[tac.Var, tac.Var | tac.Attribute | tac.Expr] | Bottom

    BOTTOM: ClassVar[Bottom] = Bottom()

    @staticmethod
    def name() -> str:
        return "Alias"

    @classmethod
    def view(cls, cfg: gu.Cfg[T]) -> IterationStrategy[T]:
        return ForwardIterationStrategy(cfg)

    def __init__(self, alias: dict[tac.Var, tac.Var] | Bottom) -> None:
        super().__init__()
        if alias is AliasDomain.BOTTOM:
            self.alias = AliasDomain.BOTTOM
        else:
            self.alias = alias.copy()

    def __le__(self, other):
        return self.join(other).alias == other.alias

    def __eq__(self, other):
        return self.alias == other.alias

    def copy(self: T) -> T:
        return AliasDomain(self.alias)

    @classmethod
    def initial(cls: Type[T]) -> T:
        return cls.top()

    @classmethod
    def top(cls: Type[T]) -> T:
        return AliasDomain({})

    @classmethod
    def bottom(cls: Type[T]) -> T:
        return AliasDomain(AliasDomain.BOTTOM)

    @property
    def is_bottom(self) -> bool:
        return self.alias == AliasDomain.BOTTOM

    def join(self: T, other: T) -> T:
        if self.is_bottom:
            return other.copy()
        if other.is_bottom:
            return self.copy()
        return AliasDomain(dict(self.alias.items() & other.alias.items()))

    def transfer(self, ins: tac.Tac, location: str) -> None:
        if self.is_bottom:
            return
        self.alias = {stackvar: v for stackvar, v in self.alias.items()
                      if stackvar not in tac.gens(ins)
                      and stackvar not in tac.free_vars(ins)}
        if isinstance(ins, tac.Assign):
            if isinstance(ins.expr, (tac.Var, tac.Attribute)) and isinstance(ins.lhs, tac.Var) and ins.lhs not in tac.free_vars_expr(ins.expr):
                left, right = ins.lhs, ins.expr
                self.alias[left] = right

    def __str__(self) -> str:
        return 'Alias({})'.format(", ".join(f'{k}={v}' for k, v in self.alias.items()))

    def __repr__(self) -> str:
        return str(self)

    def keep_only_live_vars(self, alive_vars: set[tac.Var]) -> None:
        self.alias = {k: v for k, v in self.alias.items() if k in alive_vars}


def rewrite_ins(ins: tac.Tac, pre: AliasDomain) -> tac.Tac:
    def get(name):
        return pre.alias.get(name, name)
    match ins:
        case tac.Assign():
            if tac.free_vars_expr(ins.expr) & pre.alias.keys():
                expr = ins.expr
                match ins.expr:
                    case tac.Var():
                        expr = get(ins.expr)
                    case tac.Attribute():
                        expr = dataclasses.replace(ins.expr, var=get(expr.var))
                    case tac.Subscr():
                        expr = dataclasses.replace(ins.expr, var=get(expr.var), index=get(expr.index))
                    case tac.Binary():
                        expr = dataclasses.replace(ins.expr,
                                                   left=get(expr.left),
                                                   right=get(expr.right))
                    case tac.Call():
                        expr = dataclasses.replace(ins.expr,
                                                   function=get(expr.function),
                                                   args=tuple(get(arg) for arg in expr.args))
                ins = dataclasses.replace(ins, expr=expr)
            if isinstance(ins.lhs, tac.Var) and tac.free_vars_lval(ins.lhs) & pre.alias.keys():
                lhs = ins.lhs
                match lhs:
                    case tac.Var():
                        pass
                    case tuple():
                        pass
                    case tac.Attribute():
                        lhs = dataclasses.replace(lhs, var=get(lhs.var))
                    case tac.Subscr():
                        lhs = dataclasses.replace(lhs, var=get(lhs.var), index=get(lhs.index))
                ins = dataclasses.replace(ins, lhs=lhs)
            return ins
        case _: pass
    return ins


def rewrite_aliases(block: graph_utils.Block, label: int) -> None:
    invariant: AliasDomain = block.pre[AliasDomain.name()]
    if invariant.is_bottom:
        return
    for i in range(len(block)):
        block[i] = rewrite_ins(block[i], invariant)
        invariant.transfer(block[i], f'{label}.{i}')
