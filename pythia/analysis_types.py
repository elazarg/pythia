from __future__ import annotations as _

from dataclasses import replace

import typing

import pythia.type_system as ts
from pythia.type_system import TypeExpr
from pythia import tac
from pythia.tac import Predefined, UnOp
from pythia.analysis_domain import ValueLattice


def parse_annotations(this_function: str, this_module: ts.Module):
    this_signature = ts.subscr(this_module, ts.literal(this_function))
    assert isinstance(
        this_signature, ts.Overloaded
    ), f"Expected overloaded type, got {this_signature}"
    assert (
        len(this_signature.items) == 1
    ), f"Expected single signature, got {this_signature}"
    [this_signature] = this_signature.items
    annotations = {
        tac.Var(row.index.name, is_stackvar=False): row.type
        for row in this_signature.params.row_items()
        if row.index.name is not None
    }
    annotations[tac.Var("return", is_stackvar=False)] = this_signature.return_type
    return annotations


class TypeLattice(ValueLattice[TypeExpr]):
    """
    Abstract domain for type analysis with lattice operations.
    """

    def __init__(self, this_function: str, this_module: ts.Module):
        self.annotations: dict[tac.Var, TypeExpr] = parse_annotations(
            this_function, this_module
        )
        self.globals = this_module
        self.builtins = ts.resolve_static_ref(ts.Ref("builtins"))

    def annotation(self, name: tac.Var, t: str) -> TypeExpr:
        return self.annotations[name]

    def name(self) -> str:
        return "Type"

    def join(self, left: TypeExpr, right: TypeExpr) -> TypeExpr:
        if left == right:
            return left
        result = ts.join(left, right)
        return result

    def meet(self, left: TypeExpr, right: TypeExpr) -> TypeExpr:
        return ts.meet(left, right)

    def top(self) -> TypeExpr:
        return ts.TOP

    def is_bottom(self, elem: TypeExpr) -> bool:
        return elem == ts.BOTTOM

    def bottom(self) -> TypeExpr:
        return ts.BOTTOM

    def is_less_than(self, left: TypeExpr, right: TypeExpr) -> bool:
        return ts.is_subtype(left, right)

    def resolve(self, ref: TypeExpr) -> TypeExpr:
        if isinstance(ref, ts.Ref):
            if "." in ref.name:
                module, name = ref.name.split(".", 1)
                if module == self.globals.name:
                    return ts.subscr(self.globals, ts.literal(name))
            # ref = ts.resolve_static_ref(ref)
        assert isinstance(ref, (TypeExpr, ts.Module, ts.Class)), ref
        return ref

    def join_all(self, types: typing.Iterable[TypeExpr]) -> TypeExpr:
        return ts.join_all(types)

    def call(self, function: TypeExpr, args: list[TypeExpr]) -> TypeExpr:
        return ts.call(
            self.resolve(function),
            ts.typed_dict(
                [
                    ts.make_row(index, None, self.resolve(arg))
                    for index, arg in enumerate(args)
                ]
            ),
        )

    def binary(self, left: TypeExpr, right: TypeExpr, op: str) -> TypeExpr:
        result = ts.binop(left, right, op)
        if self.is_bottom(result):
            assert False, f"Cannot resolve {op} on {left} and {right}"
            # Shorthand for: "we assume that there is an implementation"
            return self.top()
        return result

    def get_unary_attribute(self, value: TypeExpr, op: UnOp) -> TypeExpr:
        return ts.get_unop(value, self.unop_to_str(op))

    def unop_to_str(self, op: UnOp) -> str:
        match op:
            case UnOp.NEG:
                return "-"
            case UnOp.NOT:
                return "not"
            case UnOp.INVERT:
                return "~"
            case UnOp.POS:
                return "+"
            case UnOp.ITER:
                return "iter"
            case UnOp.NEXT:
                return "next"
            case UnOp.YIELD_ITER:
                return "yield iter"
            case _:
                raise NotImplementedError(f"UnOp.{op.name}")

    def unary(self, value: TypeExpr, op: UnOp) -> TypeExpr:
        f = self.get_unary_attribute(value, op)
        return self.call(f, [])

    def predefined(self, name: Predefined) -> TypeExpr:
        match name:
            case Predefined.LIST:
                return ts.make_list_constructor()
            case Predefined.SET:
                return ts.make_set_constructor()
            case Predefined.TUPLE:
                return ts.make_tuple_constructor()
            case Predefined.SLICE:
                return ts.make_slice_constructor()
            case Predefined.GLOBALS:
                return self.globals
            case Predefined.NONLOCALS:
                return self.top()
            case Predefined.LOCALS:
                return self.top()
            case Predefined.CONST_KEY_MAP:
                return self.top()
        assert False, name

    def const(self, value: object) -> TypeExpr:
        return ts.literal(value)

    def attribute(self, t: TypeExpr, attr: tac.Var) -> TypeExpr:
        mod = self.resolve(t)
        assert mod != ts.TOP, f"Cannot resolve {attr} in {t}"
        try:
            # FIX: How to differentiate nonexistent attributes from attributes that are TOP?
            res = ts.subscr(mod, ts.literal(attr.name))
            match mod, res:
                case ts.Ref(name=modname), ts.Instantiation(
                    ts.Ref("builtins.type"), (ts.Class(),)
                ):
                    arg = ts.Ref(f"{modname}.{attr.name}")
                    return replace(res, type_args=(arg,))
            if self.is_bottom(res):
                if mod == self.globals:
                    return ts.subscr(self.builtins, ts.literal(attr.name))
            return res
        except TypeError:
            if mod == self.globals:
                return ts.subscr(self.builtins, ts.literal(attr.name))
            raise

    def subscr(self, array: TypeExpr, index: TypeExpr) -> TypeExpr:
        return ts.subscr(self.resolve(array), self.resolve(index))

    def imported(self, modname: str) -> TypeExpr:
        assert isinstance(modname, str)
        return ts.resolve_static_ref(ts.Ref(modname))


def main():
    import sys
    from collections import defaultdict
    from pythia import analysis
    from pythia import disassemble
    from pythia import tac

    filename = sys.argv[1]
    function_name = sys.argv[2]

    functions, imports = disassemble.read_file(filename)
    module_type = ts.parse_file(filename)

    f = functions[function_name]
    cfg = analysis.make_tac_cfg(f, simplify=False)
    annotations = {tac.Var(k): v for k, v in f.__annotations__.items()}
    liveness_invariants = analysis.analyze(
        cfg, analysis.LivenessVarLattice(), annotations
    )
    type_analysis = analysis.domain.VarLattice(
        TypeLattice(function_name, module_type), liveness_invariants.post
    )
    type_invariants = analysis.analyze(cfg, type_analysis, annotations)
    invariant_pairs = {
        "Type": type_invariants,
    }
    analysis.print_analysis(
        cfg, invariant_pairs, defaultdict(lambda: analysis.AllocationType.NONE)
    )
