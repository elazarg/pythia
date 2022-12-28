# Data flow analysis and stuff.

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import TypeAlias, Callable, Final

import tac
from tac_analysis_domain import VarLattice, InstructionLattice, InvariantMap
from tac_analysis_types import TypeLattice, AllocationType
from type_system import TypeExpr


@dataclass(frozen=True)
class Object:
    location: str

    def __str__(self):
        return f'@{self.location}'

    def __repr__(self):
        return f'@{self.location}'


LOCALS: Final[Object] = Object('locals()')
NONLOCALS: Final[Object] = Object('NONLOCALS')

GLOBALS: Final[Object] = Object('globals()')


Graph: TypeAlias = dict[Object, dict[tac.Var, frozenset[Object]]]


def pretty_print_pointers(pointers: Graph) -> str:
    join = lambda target_obj: "{" + ", ".join(str(x) for x in target_obj) + "}"
    return ', '.join((f'{source_obj}:' if source_obj is not LOCALS else '') + f'{field}->{join(target_obj)}'
                     for source_obj in pointers
                     for field, target_obj in pointers[source_obj].items()
                     if pointers[source_obj][field]
                     )


def copy_graph(graph: Graph) -> Graph:
    return {obj: {field: target_obj.copy() for field, target_obj in obj_fields.items()}
            for obj, obj_fields in graph.items()}


class PointerLattice(InstructionLattice[Graph]):
    type_invariant_map: InvariantMap[TypeLattice]
    type_lattice: VarLattice[TypeExpr]
    liveness_analysis: InvariantMap[set[tac.Var]]
    backward: bool = False

    def name(self) -> str:
        return "Pointer"

    def __init__(self, type_invariant_map: InvariantMap[TypeLattice], type_lattice: VarLattice[TypeExpr],
                 liveness_analysis: InvariantMap) -> None:
        super().__init__()
        self.type_invariant_map = type_invariant_map
        self.type_lattice = type_lattice
        self.liveness_analysis = liveness_analysis
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
                    pointers[obj][field] = pointers[obj].get(field, frozenset()) | values
            else:
                pointers[obj] = {field: targets.copy() for field, targets in fields.items() if targets}
        return pointers

    def transfer(self, values: Graph, ins: tac.Tac, location: tuple[int, int]) -> Graph:
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

    def keep_only_live_vars(self, pointers: Graph, alive_vars: set[tac.Var]) -> None:
        for var in pointers[LOCALS].keys() - alive_vars:
            del pointers[LOCALS][var]

    def evaluator(self, state: Graph, location: tuple[int, int]) -> Callable[[tac.Expr], frozenset[Object]]:
        location_object = Object(f'{location[0]}.{location[1]}')
        locals_state = state[LOCALS]
        type_invariant = self.type_invariant_map[location]

        def inner(expr: tac.Expr) -> frozenset[Object]:
            match expr:
                case tac.Const(): return frozenset()
                case tac.Predefined(): return frozenset()
                case tac.Var(): return locals_state.get(expr, frozenset()).copy()
                case tac.Attribute():
                    attr_type = self.type_lattice.transformer_expr(type_invariant, expr.var)
                    allocation = self.type_lattice.lattice.allocation_type_attribute(attr_type, expr.field)
                    if allocation is not AllocationType.NONE:
                        return frozenset({location_object})
                    if expr.var.name == 'GLOBALS':
                        return state[GLOBALS].get(expr.field, frozenset()).copy()
                    else:
                        return frozenset(chain.from_iterable(state.get(obj, {}).get(expr.field, frozenset()).copy()
                                                             for obj in inner(expr.var)))
                case tac.Call(func, args):
                    func_type = self.type_lattice.transformer_expr(type_invariant, func)
                    allocation = self.type_lattice.lattice.allocation_type_function(func_type)
                    if allocation is AllocationType.NONE:
                        return frozenset()
                    return frozenset({location_object})
                case tac.Subscript(): return frozenset()
                case tac.Yield(): return frozenset()
                case tac.Import(): return frozenset()
                case tac.Unary():
                    value = self.type_lattice.transformer_expr(type_invariant, expr.var)
                    allocation = self.type_lattice.lattice.allocation_type_unary(value, expr.op)
                    if allocation is AllocationType.NONE:  # self.analysis.allocation_type_binary(expr.function):
                        return frozenset()
                    return frozenset({location_object})
                case tac.Binary():
                    left = self.type_lattice.transformer_expr(type_invariant, expr.left)
                    right = self.type_lattice.transformer_expr(type_invariant, expr.right)
                    allocation = self.type_lattice.lattice.allocation_type_binary(left, right, expr.op)
                    if allocation is AllocationType.NONE:  # self.analysis.allocation_type_binary(expr.function):
                        return frozenset()
                    return frozenset({location_object})
                case tac.MakeFunction(): return frozenset()
                case _: raise Exception(f"Unsupported expression {expr}")
        return inner



def allocation_to_str(t: AllocationType) -> str:
    if t is not AllocationType.NONE:
        return f' #  ' + t.name
    return ''


def mark_reachable(ptr: Graph, alive: set[tac.Var], annotations: dict[tac.Var, object], get_ins: Callable[[str, str], tac.Assign]) -> None:
    worklist = {LOCALS}
    while worklist:
        root = worklist.pop()
        for edge, locs in ptr.get(root, {}).items():
            if root == LOCALS and edge not in alive:
                continue
            if edge in annotations:
                continue
            for loc in locs:
                if repr(loc).startswith('@param'):
                    continue
                worklist.add(loc)
                label, index = [int(x) for x in str(loc)[1:].split('.')]
                ins = get_ins(label, index)
                ins.expr.allocation = AllocationType.HEAP
