# Data flow analysis and stuff.

from __future__ import annotations

from itertools import chain
from typing import TypeVar, TypeAlias, Generic, Iterator

import tac
from tac_analysis_domain import IterationStrategy, ForwardIterationStrategy, Lattice, Bottom, Top, \
    Object, LOCALS, TOP, BOTTOM, BackwardIterationStrategy, Map, Analysis, GLOBALS

import graph_utils as gu

T = TypeVar('T')
Graph: TypeAlias = [Object, Map[tac.Var, frozenset[Object]]]


class PointerGraph(Generic[T]):
    graph: Graph

    @staticmethod
    def make_map(d: dict[tac.Var, frozenset[Object]]) -> Map[tac.Var, frozenset[Object]]:
        return Map(frozenset(), d)

    def __init__(self, graph: Graph):
        assert isinstance(graph, dict)
        for obj, m in graph.items():
            assert isinstance(obj, Object)
            assert isinstance(m, Map)
            for field, targets in m.items():
                assert isinstance(field, tac.Var)
                assert isinstance(targets, frozenset|Top)
                if isinstance(targets, frozenset):
                    for target in targets:
                        assert isinstance(target, Object)
        self.graph = graph
        if GLOBALS not in self.graph:
            self.graph[GLOBALS] = PointerGraph.make_map({})

    def copy(self):
        return PointerGraph({obj: PointerGraph.make_map({field: targets.copy() for field, targets in fields.items() if targets})
                             for obj, fields in self.graph.items()})

    @staticmethod
    def join(left: PointerGraph, right: PointerGraph) -> PointerGraph:
        pointers = left.copy()
        for obj, fields in right.graph.items():
            if obj in pointers.graph:
                for field, values in fields.items():
                    pointers.graph[obj][field] = pointers.graph[obj][field] | values
            else:
                pointers.graph[obj] = {field: targets.copy() for field, targets in fields.items() if targets}
        return pointers

    def __iter__(self):
        return iter(self.graph)

    def __getitem__(self, pair: tuple[Object, tac.Var]) -> frozenset[Object]:
        obj, field = pair
        return self.graph[obj][field]

    def __setitem__(self, pair: tuple[Object, tac.Var], targets: frozenset[Object]):
        obj, field = pair
        assert isinstance(obj, Object)
        assert isinstance(field, tac.Var)
        assert isinstance(targets, frozenset|Top)
        self.graph.setdefault(obj, self.make_map({}))[field] = targets

    def __str__(self) -> str:
        return 'Pointers(' + ', '.join(f'{source_obj.pretty(field)}->{target_obj}'
                                       for source_obj in self.graph
                                       for field, target_obj in self.graph[source_obj].items()) + ")"

    def __eq__(self, other):
        if other == BOTTOM:
            return False
        if other == TOP:
            return False
        return self.graph == other.graph

    def __le__(self):
        raise NotImplementedError()

    def update_obj(self, obj, field, value) -> None:
        self[obj, field] = frozenset({self[obj, field], value})

    def update(self, pointer: PointerGraph) -> None:
        for source_obj, fields in pointer.graph.items():
            for field, targets in fields.items():
                self[source_obj, field] = targets


PointerDomain: TypeAlias = PointerGraph | Bottom | Top


class PointerLattice(Analysis[T]):
    lattice: Lattice[T]
    backward: bool

    # TODO: this does not belong here
    def view(self, cfg: gu.Cfg[T]) -> IterationStrategy:
        if self.backward:
            return BackwardIterationStrategy(cfg)
        return ForwardIterationStrategy(cfg)

    def __init__(self, lattice: Lattice[T], backward: bool = False):
        super().__init__()
        self.lattice = lattice
        self.backward = backward

    def name(self) -> str:
        return "Pointer"

    def equals(self, left: PointerDomain, right: PointerDomain) -> bool:
        return left == right

    def is_less_than(self, left: T, right: T) -> bool:
        return self.join(left, right) == right

    def is_equivalent(self, left, right) -> bool:
        return self.is_less_than(left, right) and self.is_less_than(right, left)

    def is_bottom(self, values) -> bool:
        return isinstance(values, Bottom)

    def initial(self, annotations: dict[tac.Var, str]) -> PointerDomain[T]:
        result = PointerGraph({LOCALS: PointerGraph.make_map({var: frozenset({TOP}) for var in annotations})})
        return result

    def top(self) -> PointerDomain[T]:
        return TOP

    def bottom(self) -> PointerDomain[T]:
        return BOTTOM

    def join(self, left: PointerDomain, right: PointerDomain) -> PointerDomain:
        match left, right:
            case (Top(), _): return TOP
            case (_, Top()): return TOP
            case (Bottom(), _): return right
            case (_, Bottom()): return left
            case (PointerGraph() as left, PointerGraph() as right):
                return PointerGraph.join(left, right)
        assert False, f'Unhandled case: {left} {right}'

    def meet(self, left: PointerDomain, right: PointerDomain) -> PointerDomain:
        match left, right:
            case (Bottom(), _): return BOTTOM
            case (_, Bottom()): return BOTTOM
            case (Top(), _): return right
            case (_, Top()): return left
            case (PointerGraph() as left, PointerGraph() as right):
                # TODO: add precision
                return BOTTOM
        assert False, f'Unhandled case: {left} {right}'

    def evaluate(self, pointers: PointerDomain, location: str, expr: tac.Expr) -> frozenset[Object]:
        location_object = Object(location)
        match expr:
            case tac.Const():
                return frozenset()
            case tac.Predefined.GLOBALS: return frozenset({GLOBALS})
            case tac.Predefined.LOCALS: return frozenset({LOCALS})
            case tac.Predefined.TUPLE: return frozenset()
            case tac.Var():
                return pointers[LOCALS, expr]
            case tac.Attribute(var=var, field=field):
                possible_objects = self.evaluate(pointers, location, var)
                return frozenset(chain.from_iterable(pointers[obj, field]
                                                     for obj in possible_objects))
            case tac.Subscript(var=var):
                return self.evaluate(pointers, location, tac.Attribute(var=var, field=tac.Var('*')))
            case tac.Call(function=function):
                allocation = self.lattice.is_allocation_function(self.evaluate(pointers, location, function))
                if allocation is None:
                    allocation = function.name[0].isupper()
                if allocation:
                    return frozenset({location_object})
                return frozenset()
            case tac.Binary(left=left, op=op, right=right):
                allocation = self.lattice.is_allocation_binary(self.lattice.var(left), self.lattice.var(right), op)
                if allocation is None:
                    return TOP
                elif allocation:
                    return frozenset({location_object})
                return frozenset()
            case tac.Yield():
                return TOP
            case tac.Import():
                return frozenset()
            case tac.MakeFunction():
                return frozenset()
            case _:
                raise Exception(f"Unsupported expression {expr}")

    def transformer_signature(self, pointers: PointerDomain, value: T, signature: tac.Signature) -> PointerGraph[T]:
        UNKNOWN_INDEX = tac.Var('*')
        match signature:
            case tuple():
                value_tuple = self.lattice.assign_tuple(value)
                result = PointerGraph({})
                for i in range(len(value_tuple)):
                    to_update = self.transformer_signature(pointers, value_tuple[i], signature[i])
                    result = PointerGraph.join(result, to_update)
                return result
            case tac.Var() as var:
                return PointerGraph({LOCALS: PointerGraph.make_map({var: value})})
            case tac.Attribute() as attr:
                result = PointerGraph({})
                potential_objects = self.evaluate(pointers, "", attr.var)
                if len(potential_objects) == 1:
                    # Strong update
                    [obj] = potential_objects
                    result[obj, attr.field] = value
                else:
                    # Weak update
                    for obj in potential_objects:
                        result.update_obj(obj, attr.field, value)
                return result
            case tac.Subscript(var=var):
                result = PointerGraph({})
                potential_objects = self.evaluate(pointers, "", var)
                # Weak update
                for obj in potential_objects:
                    result.update_obj(obj, UNKNOWN_INDEX, value)
                return result
            case _:
                assert False, f'unexpected signature {signature}'

    def transfer(self, pointers: PointerDomain, ins: tac.Tac, location: str) -> PointerDomain:
        pointers = pointers.copy()
        if isinstance(ins, tac.InplaceBinary):
            ins = tac.Assign(ins.lhs, tac.Binary(ins.lhs, ins.op, ins.right))
        match pointers:
            case Top(): return TOP
            case Bottom(): return TOP
            case PointerGraph() as pointers:
                match ins:
                    case tac.Assign():
                        val = self.evaluate(pointers, location, ins.expr)
                        assert isinstance(val, frozenset|Top)
                        to_update = self.transformer_signature(pointers, val, ins.lhs)
                        pointers.update(to_update)
                    case tac.InplaceBinary():
                        assert False
                    case tac.Return():
                        return pointers
                    case tac.Del():
                        # TODO
                        return pointers
                    case tac.Raise():
                        # TODO
                        return pointers
                    case tac.Jump():
                        return pointers
                    case _:
                        assert False, f'unexpected instruction {ins}'
                return pointers
            case _:
                assert False, f'unexpected pointers {pointers}'
