from __future__ import annotations as _

import typing
from copy import deepcopy
from dataclasses import dataclass

from pythia import graph_utils as gu
from pythia import tac
from pythia.graph_utils import Label, Location

T = typing.TypeVar('T')
K = typing.TypeVar('K')
Q = typing.TypeVar('Q')


@dataclass
class ForwardIterationStrategy(typing.Generic[T]):
    cfg: gu.Cfg[T]

    @property
    def entry_label(self) -> Label:
        return self.cfg.entry_label

    def successors(self, label: Label) -> typing.Iterator[Label]:
        return self.cfg.successors(label)

    def __getitem__(self, label: Label) -> gu.Block:
        return self.cfg[label]

    def order(self, pair: tuple[K, Q]) -> tuple[K ,Q]:
        return pair


@dataclass
class BackwardIterationStrategy(typing.Generic[T]):
    cfg: gu.Cfg[T]

    @property
    def entry_label(self) -> Label:
        return self.cfg.exit_label

    def successors(self, label: Label) -> typing.Iterator[Label]:
        return self.cfg.predecessors(label)

    def __getitem__(self, label: Label) -> gu.BackwardBlock:
        return typing.cast(gu.BackwardBlock, reversed(self.cfg[label]))

    def order(self, pair: tuple[K, Q]) -> tuple[Q, K]:
        return pair[1], pair[0]


IterationStrategy: typing.TypeAlias = ForwardIterationStrategy | BackwardIterationStrategy


class Lattice(typing.Protocol[T]):

    def name(self) -> str:
        raise NotImplementedError

    def top(self) -> T:
        raise NotImplementedError

    def bottom(self) -> T:
        raise NotImplementedError

    def is_top(self, elem: T) -> bool:
        raise NotImplementedError

    def is_bottom(self, elem: T) -> bool:
        raise NotImplementedError

    def join(self, left: T, right: T) -> T:
        raise NotImplementedError

    def is_less_than(self, left: T, right: T) -> bool:
        raise NotImplementedError

    def copy(self, values: T) -> T:
        raise NotImplementedError


@dataclass(frozen=True)
class Top:
    def __str__(self) -> str:
        return '⊤'

    def copy(self: T) -> T:
        return self

    def __or__(self, other: object) -> Top:
        return self

    def __ror__(self, other: object) -> Top:
        return self

    def __and__(self, other: T) -> T:
        return other

    def __rand__(self, other: T) -> T:
        return other


@dataclass(frozen=True)
class Bottom:
    def __str__(self) -> str:
        return '⊥'

    def copy(self: T) -> T:
        return self

    def __or__(self, other: T) -> T:
        return other

    def __ror__(self, other: T) -> T:
        return other

    def __rand__(self, other: object) -> Bottom:
        return self

    def __and__(self, other: object) -> Bottom:
        return self


BOTTOM = Bottom()
TOP = Top()


class Map(typing.Generic[K, T]):
    # Essentially a defaultdict, but a defaultdict makes values appear out of nowhere
    _map: dict[K, T]
    default: T

    def __init__(self, default: T, d: typing.Optional[dict[K, T]] = None):
        self.default = default
        self._map = {}
        if d is not None:
            self.update(d)

    def __getitem__(self, key: K) -> T:
        return self._map.get(key, self.default)

    def __setitem__(self, key: K, value: T) -> None:
        if value == self.default:
            if key in self._map:
                del self._map[key]
        else:
            assert not isinstance(value, dict)
            assert not isinstance(value, Map)
            self._map[key] = value

    def update(self, dictionary: dict[K, T] | Map) -> None:
        for k, v in dictionary.items():
            self[k] = v

    def __iter__(self) -> typing.Iterator[K]:
        return iter(self._map)

    def __contains__(self, key: K) -> bool:
        return key in self._map

    def __len__(self) -> int:
        return len(self._map)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Map) and self._map == other._map

    def __delitem__(self, key: K) -> None:
        del self._map[key]

    def __repr__(self) -> str:
        items = ', '.join(f'{k}={v}' for k, v in self._map.items())
        return f'Map({items})'

    def __str__(self) -> str:
        return repr(self)

    def items(self) -> typing.Iterable[tuple[K, T]]:
        return self._map.items()

    def values(self) -> typing.Iterable[T]:
        return self._map.values()

    def keys(self) -> set[K]:
        return set(self._map.keys())

    def copy(self) -> Map:
        return deepcopy(self)


MapDomain: typing.TypeAlias = Map[K, T] | Bottom
VarMapDomain: typing.TypeAlias = MapDomain[tac.Var, T]


def normalize(values: VarMapDomain[T]) -> VarMapDomain[T]:
    if isinstance(values, Bottom):
        return BOTTOM
    if any(isinstance(v, Bottom) for v in values.values()):
        return BOTTOM
    return values


class InstructionLattice(Lattice[T], typing.Protocol[T]):
    backward: bool

    def transfer(self, values: T, ins: tac.Tac, location: Location) -> T:
        raise NotImplementedError

    def initial(self, annotations: dict[tac.Var, str]) -> T:
        return self.top()


class ValueLattice(Lattice[T], typing.Protocol[T]):
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
        return self.top()

    def predefined(self, name: tac.Predefined) -> T:
        return self.top()

    def imported(self, modname: str) -> T:
        return self.top()

    def annotation(self, name: tac.Var, t: str) -> T:
        return self.top()

    def default(self) -> T:
        return self.top()


InvariantMap: typing.TypeAlias = dict[Location, T]


class VarLattice(InstructionLattice[VarMapDomain[T]], typing.Generic[T]):
    lattice: ValueLattice[T]
    liveness: InvariantMap[VarMapDomain[Top | Bottom]]
    backward: bool = False

    def __init__(self, lattice: ValueLattice[T], liveness: InvariantMap[VarMapDomain[Top | Bottom]]):
        super().__init__()
        self.lattice = lattice
        self.liveness = liveness

    def name(self) -> str:
        return f"{self.lattice.name()}"

    def is_less_than(self, left: VarMapDomain[T], right: VarMapDomain[T]) -> bool:
        if self.is_bottom(left):
            return True
        if self.is_bottom(right):
            return False
        return all(self.lattice.is_less_than(left[k], right[k]) for k in left.keys())
        # return self.join(left, right) == right

    def copy(self, values: VarMapDomain[T]) -> VarMapDomain[T]:
        return values.copy()

    def is_bottom(self, values: VarMapDomain[T]) -> bool:
        return isinstance(values, Bottom)

    def make_map(self, d: typing.Optional[dict[tac.Var, T]] = None) -> Map[tac.Var, T]:
        d = d or {}
        return Map(default=self.lattice.default(), d=d)

    def initial(self, annotations: dict[tac.Var, str]) -> VarMapDomain[T]:
        result = self.make_map()
        result.update({
            name: self.lattice.annotation(name, t)
            for name, t in annotations.items()
        })
        return result

    def top(self) -> Map[tac.Var, T]:
        return self.make_map()

    def is_top(self, elem: VarMapDomain[T]) -> bool:
        return elem == self.top()

    def bottom(self) -> VarMapDomain[T]:
        return BOTTOM

    def join(self, left: VarMapDomain[T], right: VarMapDomain[T]) -> VarMapDomain[T]:
        match left, right:
            case (Bottom(), _): return right
            case (_, Bottom()): return left
            case (Map() as left, Map() as right):
                res: Map[tac.Var, T] = self.top()
                for k in left.keys() | right.keys():
                    res[k] = self.lattice.join(left[k], right[k])
                return normalize(res)
        return self.top()

    def transformer_expr(self, values: Map[tac.Var, T], expr: tac.Expr) -> T:
        def eval(expr: tac.Var | tac.Predefined) -> T:
            res = self.transformer_expr(values, expr)
            assert not self.lattice.is_bottom(res)
            return res

        match expr:
            case tac.Var():
                res = self.lattice.var(values[expr])
                assert not self.lattice.is_bottom(res)
                return res
            case tac.Attribute():
                val = eval(expr.var)
                return self.lattice.attribute(val, expr.field)
            case tac.Call(function=function, args=args):
                func = eval(function)
                arg_values: list[T] = [eval(arg) for arg in args]
                return self.lattice.call(function=func, args=arg_values)
            case tac.Unary(var=var, op=op):
                value = eval(var)
                assert value is not None
                return self.lattice.unary(value=value, op=op)
            case tac.Binary() as expr:
                left = eval(expr.left)
                right = eval(expr.right)
                return self.lattice.binary(left=left, right=right, op=expr.op)
            case tac.Predefined() as expr:
                return self.lattice.predefined(expr)
            case tac.Const(value=value):
                return self.lattice.const(value)
            case tac.Subscript(var=var, index=index):
                return self.lattice.subscr(eval(var), eval(index))
            case tac.Yield():
                return self.lattice.top()
            case tac.Import():
                if isinstance(expr.modname, tac.Attribute):
                    val = eval(expr.modname.var)
                    return self.lattice.attribute(val, expr.modname.field)
                else:
                    return self.lattice.imported(expr.modname.name)
            case tac.MakeFunction():
                return self.lattice.top()
            case _:
                assert False, f'unexpected expr of type {type(expr)}: {expr}'

    def transformer_signature(self, value: T, signature: tac.Signature) -> Map[tac.Var, T]:
        match signature:
            case tuple() as signature:  # type: ignore
                assert all(isinstance(v, tac.Var) for v in signature)
                return self.make_map({var: self.lattice.subscr(value, self.lattice.const(i))
                                      for i, var in enumerate(signature)})
            case tac.Var():
                return self.make_map({signature: value})
            case tac.Attribute():
                return self.make_map()
            case tac.Subscript():
                return self.make_map()
            case _:
                assert False, f'unexpected signature {signature}'

    def forward_transfer(self, values: VarMapDomain[T], ins: tac.Tac) -> VarMapDomain[T]:
        if isinstance(values, Bottom):
            return BOTTOM
        if isinstance(ins, tac.For):
            ins = ins.as_call()
        to_update = self.make_map()

        match ins:
            case tac.Assign():
                assigned = self.transformer_expr(values, ins.expr)
                to_update = self.transformer_signature(assigned, ins.lhs)
            case tac.Return():
                assigned = self.transformer_expr(values, ins.value)
                to_update = self.make_map({
                    tac.Var('return'): assigned
                })
        assert not any(self.lattice.is_bottom(v) for v in to_update.values())
        return to_update

    def transfer(self, values: VarMapDomain[T], ins: tac.Tac, location: Location) -> VarMapDomain[T]:
        if isinstance(values, Bottom):
            return BOTTOM
        values = values.copy()
        try:
            print(f'values: {values}')
            print(f'{location}: {ins}')
            if location == (215, 0):
                pass
            to_update = self.forward_transfer(values, ins)
            print(f'updated: {to_update}')
            print()
        except Exception as e:
            e.add_note(f'while processing {ins} at {location}')
            raise
        for var in tac.gens(ins):
            if var in values:
                del values[var]
        if isinstance(to_update, Map):
            values.update(to_update)

        here = self.liveness[location]
        if not isinstance(here, Bottom):
            for var in set(values.keys()):
                if var.is_stackvar and here[var] is BOTTOM:
                    del values[var]
        return normalize(values)
