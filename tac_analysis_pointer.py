# Data flow analysis and stuff.

from __future__ import annotations

from itertools import chain
from typing import TypeAlias, Callable

import tac
from tac_analysis_domain import Object, LOCALS, Map, Analysis, GLOBALS, VarAnalysis

Graph: TypeAlias = dict[Object, dict[tac.Var, frozenset[Object]]]


def pretty_print_pointers(pointers) -> str:
    return ', '.join(f'{field}->{target_obj}'
                     for source_obj in pointers
                     for field, target_obj in pointers[source_obj].items()
                     if pointers[source_obj][field]
                     )


def copy_graph(graph: Graph) -> Graph:
    return {obj: {field: set(target_obj) for field, target_obj in obj_fields.items()}
            for obj, obj_fields in graph.items()}


class PointerAnalysis(Analysis[Graph]):
    backward: bool = False

    def name(self) -> str:
        return "Pointer"

    def __init__(self, analysis: VarAnalysis) -> None:
        super().__init__()
        self.analysis = analysis
        self.backward = False

    def is_less_than(self, left, right) -> bool:
        return self.join(left, right) == right

    def is_equivalent(self, left, right):
        return left == right

    def copy(self, values) -> Graph:
        return {obj: {field: targets.copy() for field, targets in fields.items() if targets}
                      for obj, fields in values.items()}

    def initial(self, annotations: dict[tac.Var, str]) -> Graph:
        return {LOCALS: {k: frozenset({Object(f'param {k}')}) for k in annotations},
                GLOBALS: {}}

    def bottom(self) -> Graph:
        return {}

    def join(self, left: Graph, right: Graph) -> Graph:
        pointers = copy_graph(left)
        for obj, fields in right.items():
            if obj in pointers:
                for field, values in fields.items():
                    pointers[obj][field] = pointers[obj].get(field, set()) | values
            else:
                pointers[obj] = {field: targets.copy() for field, targets in fields.items() if targets}
        return pointers

    def transfer(self, values: Graph, ins: tac.Tac, location: str) -> Graph:
        values = copy_graph(values)
        eval = self.evaluator(values, location)
        activation = values[LOCALS]

        for var in tac.gens(ins):
            if var in activation:
                del activation[var]

        match ins:
            case tac.Assign():
                val = eval(ins.expr)
                match ins.lhs:
                    case tac.Var():
                        activation[ins.lhs] = val
                    case tac.Attribute():
                        for obj in eval(ins.lhs.var):
                            values.setdefault(obj, {})[ins.lhs.field] = val
                    case tac.Subscript():
                        for obj in eval(ins.lhs.var):
                            values.setdefault(obj, {})[tac.Var('*')] = val
            case tac.Return():
                val = eval(ins.value)
                activation[tac.Var('return')] = val
        return values

    def to_string(self, pointers) -> str:
        return 'Pointers(' + ', '.join(f'{source_obj.pretty(field)}->{target_obj}'
                                       for source_obj in pointers
                                       for field, target_obj in pointers[source_obj].items()) + ")"

    def keep_only_live_vars(self, pointers, alive_vars: set[tac.Var]) -> None:
        for var in pointers[LOCALS].keys() - alive_vars:
            del pointers[LOCALS][var]

    def evaluator(self, state: Graph, location: str) -> Callable[[tac.Expr], frozenset[Object]]:
        location_object = Object(location)
        locals_state = state[LOCALS]

        def inner(expr: tac.Expr) -> frozenset[Object]:
            match expr:
                case tac.Const(): return frozenset()
                case tac.Predefined(): return frozenset()
                case tac.Var(): return locals_state.get(expr, frozenset()).copy()
                case tac.Attribute():
                    if expr.is_allocation:
                        return frozenset({location_object})
                    if expr.var.name == 'GLOBALS':
                        return state[GLOBALS].get(expr.field, frozenset()).copy()
                    else:
                        return frozenset(chain.from_iterable(state.get(obj, {}).get(expr.field, frozenset()).copy()
                                                       for obj in inner(expr.var)))
                case tac.Call():
                    if not expr.is_allocation:
                        return frozenset()
                    return frozenset({location_object})
                case tac.Subscript(): return frozenset()
                case tac.Yield(): return frozenset()
                case tac.Import(): return frozenset()
                case tac.Binary():
                    if not expr.is_allocation:  # self.analysis.is_allocation_binary(expr.function):
                        return frozenset()
                    return frozenset({location_object})
                case tac.MakeFunction(): return frozenset()
                case _: raise Exception(f"Unsupported expression {expr}")
        return inner
