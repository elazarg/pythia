from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path
import ast


class TypeExpr:
    def __and__(self, other):
        if isinstance(other, Intersection):
            return other & self
        return Intersection(frozenset({self, other}))

T = typing.TypeVar('T', bound=TypeExpr)
K = typing.TypeVar('K')


@dataclass(frozen=True)
class Literal(TypeExpr, typing.Generic[K]):
    value: K


@dataclass(frozen=True)
class Infer(TypeExpr):
    pass


@dataclass(frozen=True)
class Ref(TypeExpr):
    name: str
    static: bool = True

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class TypeVar(TypeExpr):
    name: str

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class Row(TypeExpr):
    index: typing.Optional[int]
    name: typing.Optional[str]
    type: TypeExpr

    def __repr__(self):
        return f'{self.name}: {self.type}'

    def match(self, index_or_name: Literal[int] | Literal[str]):
        return index_or_name.value == self.index or index_or_name.value == self.name


R = typing.TypeVar('R', bound=Row)


@dataclass(frozen=True)
class Intersection(TypeExpr, typing.Generic[R]):
    items: frozenset[R]

    def __repr__(self) -> str:
        return f'{{{" & ".join(f"{decl}" for decl in self.items)}}}'

    def __and__(self, other: TypeExpr) -> Intersection[R]:
        if not isinstance(other, Intersection):
            raise TypeError(f'Cannot intersect {self} with {other}')
        if not self.items:
            return other
        if not other.items:
            return self
        res = Intersection(self.items | other.items)
        assert len(set(type(item) for item in res.items)) <= 1
        return res

    def split_by_index(self: Intersection[Row], index: Literal[str] | Literal[int]) -> tuple[Intersection[Row], Intersection[Row]]:
        good = intersect([item for item in self.items if item.match(index)])
        bad = intersect([item for item in self.items if not item.match(index)])
        return good, bad


TypedDict: typing.TypeAlias = Intersection[Row]

@dataclass(frozen=True)
class Union(TypeExpr):
    items: frozenset[TypeExpr]


@dataclass(frozen=True)
class Class:
    name: str
    class_dict: TypedDict
    inherits: tuple[TypeExpr, ...] = ()

    def __repr__(self):
        return f'class {self.name}({self.class_dict}) < {self.inherits}'


@dataclass(frozen=True)
class Module:
    name: str
    class_dict: TypedDict

    def __repr__(self):
        return f'module {self.name}({self.class_dict})'


@dataclass(frozen=True)
class Protocol:
    class_dict: TypedDict
    inherits: tuple[TypeExpr, ...] = ()

    def __repr__(self):
        return f'protocol({self.inherits}): {self.class_dict}'


@dataclass(frozen=True)
class FunctionType(TypeExpr):
    params: TypedDict
    return_type: TypeExpr
    new: Literal[bool]
    property: Literal[bool] = Literal[bool](False)

    def __repr__(self) -> str:
        if self.params is None:
            return f'->{self.return_type}'
        else:
            return f'({self.params} -> {"new " if self.new else ""}{self.return_type})'

    def is_property(self) -> bool:
        return self.params is None


G = typing.TypeVar('G', Class, Protocol, FunctionType, Ref)


@dataclass(frozen=True)
class Generic(TypeExpr, typing.Generic[G]):
    type_params: tuple[TypeVar]
    type: G

    def __repr__(self) -> str:
        return f'[{", ".join(repr(x) for x in self.type_params)}]{self.type}'

    def __getitem__(self, *items: TypeExpr) -> G:
        return simplify_generic(Instantiation(self, items))


@dataclass(frozen=True)
class Instantiation(TypeExpr, typing.Generic[G]):
    generic: Generic[G]
    type_params: tuple[TypeExpr]

    def __repr__(self):
        return f'{self.generic}[{", ".join(repr(x) for x in self.type_params)}]'


def simplify_generic(t):
    def simplify_generic(t, context: dict[TypeVar, TypeExpr]):
        match t:
            case Infer():
                return Infer()
            case Generic(type_params, typeexpr):
                new_context = {k: v for k, v in context.items() if k not in type_params}
                return Generic(type_params, simplify_generic(typeexpr, new_context))
            case Instantiation(Generic() as generic, type_params):
                new_type_params = tuple(simplify_generic(t, context) for t in type_params)
                new_context = {k: v for k, v in zip(generic.type_params, new_type_params)}
                new_context = context | new_context
                return simplify_generic(generic.type, new_context)
            case Instantiation(Intersection() as items, type_params):
                return Intersection(frozenset(simplify_generic(Instantiation(item, type_params), context) for item in items))
            case Ref():
                return t
            case TypeVar():
                return context.get(t, t)
            case Intersection(items):
                return Intersection(frozenset(simplify_generic(x, context) for x in items))
            case Row(index, name, typeexpr):
                return Row(index, name, simplify_generic(typeexpr, context))
            case FunctionType(params, return_type, new):
                new_params = simplify_generic(params, context)
                new_return_type = simplify_generic(return_type, context)
                return FunctionType(new_params, new_return_type, new)
            case _: raise NotImplementedError(f'{t!r}, {type(t)}')
    return simplify_generic(t, {})


def intersect(rows: typing.Iterable[Row]) -> TypedDict:
    return Intersection[Row](frozenset(rows))


TOP = intersect([])
BOTTOM = Union(frozenset())


def infer_self(function: FunctionType) -> Generic[FunctionType]:
    params = function.params
    if not params:
        raise TypeError(f'Cannot bind self to {function}')
    self_arg, other_args = params.split_by_index(Literal[int](0))
    assert len(self_arg.items) == 1
    self_arg = next(iter(self_arg.items))
    if self_arg.type == Infer():
        self_arg = Row(0, self_arg.name, TypeVar('Self'))
    params = intersect([self_arg]) & other_args
    return Generic((TypeVar('Self'),), FunctionType(params, function.return_type, function.new))


def expr_to_type(expr: ast.expr) -> TypeExpr:
    match expr:
        case None:
            return Infer()
        case ast.Constant(None):
            return Ref('None')
        case ast.Name(id=id):
            return Ref(id)
        case ast.Subscript(value=value, slice=ast.Tuple(elts=elts)):
            return Instantiation(expr_to_type(value), tuple(expr_to_type(x) for x in elts))

# convert a Python ast to a type expression
def def_to_type(definition: ast.stmt) -> Class | Protocol | FunctionType:
    match definition:
        case ast.ClassDef(name=name, body=body, bases=base_expressions, decorator_list=decorator_list):
            class_dict = intersect([Row(index, stmt.name, infer_self(def_to_type(stmt))) for index, stmt in enumerate(body)
                                    if isinstance(stmt, ast.FunctionDef)])
            base_classes = []
            protocol = None
            generic_params: tuple[TypeVar] = ()
            for base in base_expressions:
                match base:
                    case ast.Name(id='Protocol'):
                        protocol = True
                    case ast.Subscript(value=ast.Name(id='Protocol' | 'Generic'), slice=ast.Name(id=id)):
                        protocol = id == 'Protocol'
                        generic_params = (TypeVar(id),)
                    case ast.Subscript(value=ast.Name(id='Protocol' | 'Generic'), slice=ast.Tuple(elts=elts)):
                        protocol = id == 'Protocol'
                        generic_params = tuple([expr_to_type(id) for id in elts])
                    case ast.Name(id=id):
                        base_classes.append(Ref(id))
            base_classes = tuple(base_classes)
            if protocol is not None:
                res = Protocol(class_dict, inherits=base_classes)
            else:
                res = Class(name, class_dict, inherits=base_classes)
            if generic_params:
                res = Generic(generic_params, res)
            return res
        case ast.FunctionDef(name=name, returns=returns, decorator_list=decorator_list,
                             args=ast.arguments(args=args, vararg=vararg, kwonlyargs=kwonlyargs, kw_defaults=kw_defaults, kwarg=kwarg, defaults=defaults)):
            params = intersect([Row(index, arg.arg, expr_to_type(arg.annotation)) for index, arg in enumerate(args)])
            returns = expr_to_type(returns)
            name_decorators = [decorator.id for decorator in decorator_list if isinstance(decorator, ast.Name)]
            new = Literal[bool]('new' in name_decorators)
            property = Literal[bool]('property' in name_decorators)
            return FunctionType(params, returns, new=new, property=property)
        case _: raise NotImplementedError(f'{definition!r}, {type(definition)}')


def module_to_type(module: ast.Module, name: str) -> Module:
    class_dict = intersect([Row(index, stmt.name, def_to_type(stmt)) for index, stmt in enumerate(module.body)
                            if isinstance(stmt, (ast.FunctionDef, ast.ClassDef))])
    return Module(name, class_dict)


def pretty_print_type(t: Module | TypeExpr, indent=0):
    match t:
        case Intersection(items):
            for row in items:
                pretty_print_type(row, indent)
        case Row(index, name, typeexpr):
            print(' ' * indent, name, end='', sep='')
            if isinstance(typeexpr, Generic):
                print(f'[{", ".join(str(t) for t in typeexpr.type_params)}]', sep='', end='')
                typeexpr = typeexpr.type
            if isinstance(typeexpr, Intersection):
                print()
            pretty_print_type(typeexpr, indent + 4)
        case FunctionType(params, return_type, Literal(new), Literal(property)):
            pretty_params = ', '.join(f'{row.name}: {row.type}' for row in sorted(params.items, key=lambda x: x.index))
            print(f'({pretty_params}) -> {"new " if new else ""}{return_type}')
        case Class(name, typeexpr):
            print(f': class {name}', sep='')
            pretty_print_type(typeexpr, indent)
        case Protocol(typeexpr):
            print(f': protocol', sep='')
            pretty_print_type(typeexpr, indent)
        case Module(name, typeexpr):
            print(f'module {name}:', sep='')
            pretty_print_type(typeexpr, indent+4)
        case Generic(type_params, typeexpr):
            print(f'[{", ".join(str(t) for t in type_params)}]', sep='', end='')
            pretty_print_type(typeexpr, indent)
        case _:
            raise NotImplementedError(f'{t!r}, {type(t)}')


def parse_file(path: str) -> Module:
    with open(path) as f:
        tree = ast.parse(f.read())
    module = module_to_type(tree, Path(path).stem)
    return module


def join(t1: TypeExpr, t2: TypeExpr):
    match t1, t2:
        case Intersection(items1), Intersection(items2):
            return Intersection(items1 | items2)
        case Intersection(items), other:
            return Intersection(items | {other})
        case other, Intersection(items):
            return Intersection(items | {other})
        case Literal(value1), Literal(value2):
            if value1 == value2:
                return Literal(value1)
            else:
                return Union(frozenset({t1, t2}))
        case FunctionType(params1, return_type1, new1), FunctionType(params2, return_type2, new2):
            return FunctionType(join(params1,  params2), meet(return_type1, return_type2), meet(new1, new2))
        case _: raise NotImplementedError(f'{t1!r}, {type(t1)}')


def meet(t1: TypeExpr, t2: TypeExpr):
    if t1 == TOP:
        return t2
    if t2 == TOP:
        return t1
    match (t1, t2):
        case (Infer(), _):
            return t2
        case (_, Infer()):
            return t1
        case (Literal(value1), Literal(value2)):
            if value1 == value2:
                return Literal(value1)
            else:
                return BOTTOM
        case (Union(items1), Union(items2)):
            return Union(items1 & items2)
        case (Union(items), t) | (t, Union(items)):
            return Union(frozenset(meet(item, t) for item in items))
        case (Intersection(items1), Intersection(items2)):
            return Intersection(items1 | items2)
        case (Intersection(items), Row() as r) | (Row() as r, Intersection(items)):
            return Intersection(items | {r})
        case (Intersection(items), t) | (t, Intersection(items)):
            return TOP
        case (FunctionType(params1, return_type1, new1, property1), FunctionType(params2, return_type2, new2, property2)):
            return FunctionType(meet(params1, params2),
                                join(return_type1, return_type2),
                                join(new1, new2),
                                join(property1, property2))
        case (t1, t2):
            if t1 == t2:
                return t1
            else:
                return Union(frozenset({t1, t2}))


def meet_all(items: typing.Iterable[TypeExpr]) -> TypeExpr:
    res = TOP
    for t in items:
        res = meet(res, t)
    return res


def apply(t: TypeExpr, action: str, arg: TypeExpr) -> TypeExpr:
    match t, action, arg:
        case Intersection(items), action, arg:
            return meet_all(apply(item, action, arg) for item in items)
        case t, ('getattr' | 'getitem'), index:
            good, bad = t.class_dict.split_by_index(index)
            if not good.items:
                raise TypeError(f'No attribute {index} in {t}')
            return Intersection(frozenset({x.type for x in good.items}))
        case FunctionType(return_type=return_type), 'call', arg:
            return return_type
        case Class(), 'call', arg:
            return apply(apply(t.class_dict, 'getattr', Literal[str]('__call__')), action, arg)
        case _:
            raise NotImplementedError(f'{t!r}, {action!r}, {arg!r}')


def main():
    mod = parse_file('typeshed_mini/builtins.pyi')
    pretty_print_type(mod)
    S = apply(mod, 'getattr', Literal[str]('sum'))
    print(S)
    X = apply(S, 'call', Literal[int](1))
    print(X)


if __name__ == '__main__':
    main()


# SimpleType: TypeAlias = Ref | TypeVar | OverloadedFunctionType | MapType | Generic | Instantiation | FunctionType
#
# def instantiate(self: T, type_args: dict[TypeVar, SimpleType]) -> T: pass
#
# def bind(self: T, type_args: dict[TypeVar, SimpleType]) -> T: pass
#
#
# def make_tuple(t: ObjectType) -> ObjectType:
#     return ObjectType('tuple', frozendict({
#         '__getitem__': make_function_type(t),
#     }))
#
#
# def iter_method(element_type: ObjectType) -> OverloadedFunctionType:
#     return make_function_type(ObjectType(f'iterator', frozendict({
#         '__next__': make_function_type(element_type),
#     }), type_params=frozendict({'Iterator': element_type})), new=True)
#
#
# OBJECT = ObjectType('object')
# FLOAT = ObjectType('float')
# INT = ObjectType('int')
# STRING = ObjectType('str')
# BOOL = ObjectType('bool', frozendict({
#     '__bool__': make_function_type(Ref('bool'), new=False),
# }))
# TYPE = ObjectType('type', frozendict({
#     '__call__': make_function_type(TypeVar('T'), new=False),
# }), type_params=frozendict({'T': None}))
# NONE = ObjectType('None')
# CODE = ObjectType('code')
# ASSERTION_ERROR = ObjectType('AssertionError')
# SLICE = ObjectType('slice')
#
# LIST = ObjectType('list', frozendict({
#     '__getitem__': make_function_type(TypeVar('T'), new=False),
#     '__iter__': iter_method(TypeVar('T')),
#     '__len__': make_function_type(INT, new=False),
#     '__contains__': make_function_type(BOOL, new=False),
#     'clear': make_function_type(NONE, new=False),
#     'copy': make_function_type(NONE, new=False),
#     'count': make_function_type(INT, new=False),
#     'extend': make_function_type(NONE),
#     'index': make_function_type(INT, new=False),
#     'insert': make_function_type(NONE, new=False),
#     'pop': make_function_type(TypeVar('T')),
#     'remove': make_function_type(NONE, new=False),
#     'reverse': make_function_type(NONE, new=False),
#     'sort': make_function_type(NONE, new=False),
# }), type_params=frozendict({'T': None}))
#
# DICT = ObjectType('dict', frozendict({
#     '__getitem__': make_function_type(OBJECT, new=False),
#     '__setitem__': make_function_type(NONE, new=False),
# }))
#
# TUPLE_CONSTRUCTOR = make_function_type(make_tuple(OBJECT), new=False)
# LIST_CONSTRUCTOR = make_function_type(LIST, ParamsType(varargs=True), new=False)
# SLICE_CONSTRUCTOR = make_function_type(SLICE, new=False)
# DICT_CONSTRUCTOR = make_function_type(DICT, new=True)
#
# NDARRAY = ObjectType('ndarray', frozendict({
#     'mean': make_function_type(FLOAT, new=False, pure=True),
#     'std': make_function_type(FLOAT, new=False, pure=True),
#     'shape': make_tuple(INT),
#     'size': INT,
#     '__getitem__': OverloadedFunctionType(
#         tuple([
#             FunctionType(ParamsType((INT,), varargs=False), FLOAT, new=True),
#             FunctionType(ParamsType((FLOAT,), varargs=False), FLOAT, new=True),  # Fix: ad-hoc
#             FunctionType(ParamsType((SLICE,), varargs=False), Ref('numpy.ndarray'), new=True),
#             FunctionType(ParamsType((Ref('numpy.ndarray'),), varargs=False), Ref('numpy.ndarray'), new=True),
#             FunctionType(ParamsType((make_tuple(OBJECT),), varargs=False), OBJECT, new=True),  # FIX: not necessarily new
#         ])
#     ),
#     '__iter__': iter_method(FLOAT),  # inaccurate
#     'T': Property(Ref('numpy.ndarray'), new=True),
#     'astype': make_function_type(Ref('numpy.ndarray'), new=True, pure=True),
#     'reshape': make_function_type(Ref('numpy.ndarray'), new=True, pure=False),
#     'ndim': INT,
# }))
#
# ARRAY_GEN = make_function_type(NDARRAY, new=True)
#
# DATAFRAME = ObjectType('DataFrame', frozendict({
#     'loc': DICT,
# }))
#
# TIME_MODULE = ObjectType('/time')
#
# NUMPY_MODULE = ObjectType('/numpy', frozendict({
#     'ndarray': TYPE[NDARRAY],
#     'array': ARRAY_GEN,
#     'dot': make_function_type(FLOAT, new=False),
#     'zeros': ARRAY_GEN,
#     'ones': ARRAY_GEN,
#     'concatenate': ARRAY_GEN,
#     'empty': ARRAY_GEN,
#     'empty_like': ARRAY_GEN,
#     'full': ARRAY_GEN,
#     'full_like': ARRAY_GEN,
#     'arange': ARRAY_GEN,
#     'linspace': ARRAY_GEN,
#     'logspace': ARRAY_GEN,
#     'geomspace': ARRAY_GEN,
#     'meshgrid': ARRAY_GEN,
#     'max': make_function_type(FLOAT, new=False),
#     'min': make_function_type(FLOAT, new=False),
#     'sum': make_function_type(FLOAT, new=False),
#     'setdiff1d': ARRAY_GEN,
#     'unique': ARRAY_GEN,
#     'append': ARRAY_GEN,
#     'random': ObjectType('/numpy.random', frozendict({
#         'rand': ARRAY_GEN,
#     })),
#     'argmax': make_function_type(INT, new=False),
#     'c_': ObjectType('slice_trick', frozendict({
#         '__getitem__': make_function_type(NDARRAY),
#     })),
#     'r_': ObjectType('slice_trick', frozendict({
#         '__getitem__': make_function_type(NDARRAY),
#     })),
# }))
#
# SKLEARN_MODULE = ObjectType('/sklearn', frozendict({
#     'metrics': ObjectType('/metrics', frozendict({
#         'log_loss': make_function_type(FLOAT, new=False),
#         'accuracy_score': make_function_type(FLOAT, new=False),
#         'f1_score': make_function_type(FLOAT, new=False),
#         'precision_score': make_function_type(FLOAT, new=False),
#         'recall_score': make_function_type(FLOAT, new=False),
#         'roc_auc_score': make_function_type(FLOAT, new=False),
#         'average_precision_score': make_function_type(FLOAT, new=False),
#         'roc_curve': make_function_type(make_tuple(FLOAT), new=False),
#         'confusion_matrix': make_function_type(make_tuple(INT), new=False),
#     })),
#     'linear_model': ObjectType('/linear_model', frozendict({
#         'LogisticRegression': make_function_type(ObjectType('LogisticRegression', frozendict({
#             'fit': make_function_type(ObjectType('Model', frozendict({
#                 'predict': make_function_type(FLOAT),
#                 'predict_proba': ARRAY_GEN,
#                 'score': make_function_type(FLOAT),
#             })), new=False),
#         })), new=True),
#         'LinearRegression': make_function_type(ObjectType('LinearRegression', frozendict({
#             'fit': make_function_type(ObjectType('Model', frozendict({
#                 'predict': make_function_type(FLOAT),
#                 'predict_proba': ARRAY_GEN,
#                 'score': make_function_type(FLOAT),
#             })), new=False),
#         })), new=True),
#     })),
# }))
#
# PANDAS_MODULE = ObjectType('/pandas', frozendict({
#     'DataFrame': TYPE[DATAFRAME],
# }))
#
# BUILTINS_MODULE = ObjectType('/builtins', frozendict({
#     'range': make_function_type(ObjectType('range', frozendict({
#         '__iter__': iter_method(INT),
#     }))),
#     'zip': make_function_type(ObjectType('zip', frozendict({
#         '__iter__': iter_method(make_tuple(OBJECT)),
#     }))),
#
#     'len': make_function_type(INT, new=False),
#     'print': make_function_type(NONE, new=False),
#     'abs': make_function_type(FLOAT, new=False),
#     'round': make_function_type(FLOAT, new=False),
#     'min': make_function_type(FLOAT, new=False),
#     'max': make_function_type(FLOAT, new=False),
#     'sum': make_function_type(FLOAT, new=False),
#     'all': make_function_type(BOOL, new=False),
#     'any': make_function_type(BOOL, new=False),
#
#     'int': TYPE[INT],
#     'float': TYPE[FLOAT],
#     'str': TYPE[STRING],
#     'bool': TYPE[BOOL],
#     'code': TYPE[CODE],
#     'object': TYPE[OBJECT],
#     'AssertionError': TYPE[ASSERTION_ERROR],
# }))
#
# FUTURE_MODULE = ObjectType('/future', frozendict({
#     'annotations': ObjectType('_'),
# }))
#
# MATPLOTLIB_MODULE = ObjectType('/matplotlib', frozendict({
#     'pyplot': ObjectType('pyplot', frozendict({
#         'plot': make_function_type(NONE, new=False),
#         'show': make_function_type(NONE, new=False),
#     })),
# }))
#
# PERSIST_MODULE = ObjectType('/persist', frozendict({
#     'range': make_function_type(ObjectType('range', frozendict({
#         '__iter__': iter_method(INT),
#     }))),
# }))
#
# MODULES = frozendict({
#     '__future__': FUTURE_MODULE,
#     'numpy': NUMPY_MODULE,
#     'pandas': PANDAS_MODULE,
#     'time': TIME_MODULE,
#     'sklearn': SKLEARN_MODULE,
#     'matplotlib': MATPLOTLIB_MODULE,
#     'persist': PERSIST_MODULE,
#     'builtins': BUILTINS_MODULE,
# })
#
#
# def resolve_module(path: str, modules: dict = MODULES) -> ObjectType:
#     path = path.split('.')
#     obj = modules[path[0]]
#     for part in path[1:]:
#         obj = obj.fields[part]
#     return obj
#
#
# def transpose(double_dispatch_table: dict[tuple[SimpleType, SimpleType], dict[str, tuple[SimpleType, bool]]]
#               ) -> dict[str, OverloadedFunctionType]:
#     ops = ['+', '-', '*', '/', '**', '//', '%', '@',
#            '==', '!=', '<', '<=', '>', '>=']
#     result: dict[str, list[FunctionType]] = {op: [] for op in ops}
#     for op in ops:
#         for params, table in double_dispatch_table.items():
#             if op in table:
#                 return_type, new = table[op]
#                 result[op].append(FunctionType(ParamsType(params), return_type, new))
#     return {op: OverloadedFunctionType(tuple(result[op])) for op in ops}
#
#
# BINARY = transpose({
#     (INT, INT): {
#         '/':   (FLOAT, False),
#         **{op: (INT, False) for op in ['+', '-', '*',       '**', '//', '%']},
#         **{op: (INT, False) for op in ['&', '|', '^', '<<', '>>']},
#         **{op: (BOOL, False) for op in ['<', '>', '<=', '>=', '==', '!=']},
#     },
#     (INT, FLOAT): {
#         **{op: (FLOAT, False) for op in ['+', '-', '*', '/', '**', '//', '%']},
#         **{op: (BOOL, False) for op in ['<', '>', '<=', '>=', '==', '!=']},
#     },
#     (FLOAT, INT): {
#         **{op: (FLOAT, False) for op in ['+', '-', '*', '/', '**', '//', '%']},
#         **{op: (BOOL, False) for op in ['<', '>', '<=', '>=', '==', '!=']},
#     },
#     (FLOAT, FLOAT): {
#         **{op: (FLOAT, False) for op in ['+', '-', '*', '/', '**', '//', '%']},
#         **{op: (BOOL, False) for op in ['<', '>', '<=', '>=', '==', '!=']},
#     },
#     (NDARRAY, NDARRAY): {op: (NDARRAY, True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
#     (NDARRAY, INT):     {op: (NDARRAY, True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
#     (INT, NDARRAY):     {op: (NDARRAY, True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
#     (NDARRAY, FLOAT):   {op: (NDARRAY, True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
#     (FLOAT, NDARRAY):   {op: (NDARRAY, True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
# })
