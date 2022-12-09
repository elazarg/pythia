from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import ast


class TypeExpr:
    def __and__(self, other):
        if isinstance(other, Intersection):
            return other & self
        return Intersection(frozenset({self, other}))


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
    index: Optional[int]
    name: Optional[str]
    type: TypeExpr

    def __repr__(self):
        return f'{self.name}: {self.type}'

    def match(self, index_or_name: int | str):
        return index_or_name == self.index or index_or_name == self.name


T = typing.TypeVar('T', bound=TypeExpr)


@dataclass(frozen=True)
class Intersection(TypeExpr, typing.Generic[T]):
    items: frozenset[T]

    def __repr__(self) -> str:
        return f'{{{" & ".join(f"{decl}" for decl in self.items)}}}'

    def __and__(self, other: TypeExpr) -> Intersection[T]:
        if not isinstance(other, Intersection):
            raise TypeError(f'Cannot intersect {self} with {other}')
        if not self.items:
            return other
        if not other.items:
            return self
        res = Intersection(self.items | other.items)
        assert len(set(type(item) for item in res.items)) <= 1
        return res

    def split_by_index(self: Intersection[Row], index: str | int) -> tuple[Intersection[Row], Intersection[Row]]:
        res = intersect(*[item for item in self.items if item.match(index)])
        return res, Intersection(self.items - res.items)


@dataclass(frozen=True)
class Generic(TypeExpr, typing.Generic[T]):
    type_params: tuple[TypeVar]
    type: T

    def __repr__(self) -> str:
        return f'[{", ".join(repr(x) for x in self.type_params)}]{self.type}'

    def __getitem__(self, *items: TypeExpr) -> T:
        return simplify_generic(Instantiation(self, items))


@dataclass(frozen=True)
class Instantiation(TypeExpr, typing.Generic[T]):
    generic: Generic[T]
    type_params: tuple[TypeExpr]

    def __repr__(self):
        return f'{self.generic}[{", ".join(repr(x) for x in self.type_params)}]'


TypedDict: typing.TypeAlias = Intersection[Row]

@dataclass(frozen=True)
class Class(TypeExpr):
    name: str
    class_dict: TypedDict
    inherits: tuple[Class, ...] = ()

    def __repr__(self):
        return f'class {self.name}({self.class_dict})'


@dataclass(frozen=True)
class FunctionType(TypeExpr):
    params: Optional[TypedDict]
    return_type: TypeExpr
    new: bool

    def __repr__(self) -> str:
        if self.params is None:
            return f'->{self.return_type}'
        else:
            return f'({self.params} -> {"new " if self.new else ""}{self.return_type})'

    def is_property(self) -> bool:
        return self.params is None

    def add_self_type(self) -> FunctionType:
        self_item, other_items = self.params.split_by_index(0)
        return FunctionType(
            params=intersect(Row(0, 'self', self)) & other_items,
            return_type=self.return_type,
            new=self.new,
        )


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


def intersect(*rows: Row) -> TypedDict:
    return Intersection[Row](frozenset(rows))


TOP = intersect()


def protocol(typed_dict: TypedDict, type_params=(), implements: TypedDict = TOP) -> Generic[TypedDict]:
    m = Generic((TypeVar('Self'),), typed_dict & implements)
    if type_params:
        return Generic(tuple(TypeVar(x) for x in type_params), m)
    return m


def klass(name: str, class_dict: TypedDict, implements: TypedDict = TOP) -> Class:
    res = protocol(class_dict, implements=implements)
    return Class(name, res[Ref(name)])


def method(*rows: Row, return_type: TypeExpr, new: bool):
    return FunctionType(intersect(Row(0, 'self', TypeVar('Self')), *rows),
                        return_type=return_type,
                        new=new)

def bind_self(function: FunctionType) -> FunctionType:
    params = function.params
    if not params:
        raise TypeError(f'Cannot bind self to {function}')
    self_arg, other_args = params.split_by_index(0)
    assert len(self_arg.items) == 1
    self_arg = next(iter(self_arg.items))
    if self_arg.type == Infer():
        self_arg = Row(self_arg.index, self_arg.name, TypeVar('Self'))
    params = intersect(self_arg) & other_args
    return FunctionType(params, function.return_type, function.new)


# convert a Python ast to a type expression
def ast_to_type(tree: ast.AST):
    match tree:
        case None:
            return Infer()
        case ast.Constant(None):
            return Ref('None')
        case ast.Name(id=id):
            return Ref(id)
        case ast.Module(body):
            return intersect(*[Row(None, stmt.name, ast_to_type(stmt)) for stmt in body
                               if isinstance(stmt, (ast.FunctionDef, ast.ClassDef))])
        case ast.ClassDef(name=name, body=body, bases=bases, decorator_list=decorator_list):
            functions = [Row(i, stmt.name, bind_self(ast_to_type(stmt))) for i, stmt in enumerate(body)
                         if isinstance(stmt, ast.FunctionDef)]
            for base in bases:
                if isinstance(base, ast.Name) and base.id == 'Protocol':
                    return protocol(intersect(*functions))
                if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name) and base.value.id == 'Protocol':
                    return protocol(intersect(*functions))
            return klass(name, intersect(*functions))
        case ast.FunctionDef(name=name, returns=returns, args=ast.arguments(args=args, vararg=vararg, kwonlyargs=kwonlyargs, kw_defaults=kw_defaults, kwarg=kwarg, defaults=defaults)):
            params = intersect(*[Row(index, arg.arg, ast_to_type(arg.annotation)) for index, arg in enumerate(args)])
            returns = ast_to_type(returns)
            return FunctionType(params, returns, new=False)
        case _: raise NotImplementedError(f'{tree!r}, {type(tree)}')


def pretty_print_type(t: TypeExpr, indent=0):
    match t:
        case Intersection(items):
            for row in items:
                pretty_print_type(row, indent)
        case Row(index, name, typeexpr):
            print(' ' * indent, name, end='', sep='')
            if isinstance(typeexpr, Intersection):
                print()
            pretty_print_type(typeexpr, indent + 4)
        case FunctionType(params, return_type, new):
            print(f'({params} -> {"new " if new else ""}{return_type})')
        case Class(name, typeexpr):
            print(f': class {name}', sep='')
            pretty_print_type(typeexpr, indent)
        case Generic(type_params, typeexpr):
            print(f'[{", ".join(str(t) for t in type_params)}]', sep='')
            pretty_print_type(typeexpr, indent)


def parse_file(path: str) -> Row:
    with open(path) as f:
        tree = ast.parse(f.read())
    module = Row(None, Path(path).stem, ast_to_type(tree))
    return module


def main():
    INT = Ref('builtins.int', static=True)
    FLOAT = Ref('builtins.float', static=True)

    SIZEABLE_PROTOCOL = protocol(intersect(Row(None, '__len__', method(return_type=INT, new=False))))

    LEN = FunctionType(intersect(Row(None, 'obj', SIZEABLE_PROTOCOL)), INT, new=False)

    NEXT_METHOD = Row(None, '__next__', method(return_type=TypeVar('T'), new=False))

    ITERATOR_PROTOCOL = intersect(NEXT_METHOD)

    ITERABLE_METHOD = method(return_type=protocol(ITERATOR_PROTOCOL), new=True)

    ITERABLE_PROTOCOL = Generic((TypeVar('T'),), intersect(Row(None, '__iter__', ITERABLE_METHOD)))

    NONE = Ref('builtins.None', static=True)
    ITERABLE_INT = ITERABLE_PROTOCOL[INT]

    OBJECT_DICT = intersect(
           Row(0, '__str__', method(return_type=NONE, new=False)),
           Row(1, '__hash__', method(return_type=INT, new=False)),
    )

    PAIR = intersect(
        Row(0, None, INT),
        Row(1, None, FLOAT),
    )

    print(f'              PAIR={PAIR}')
    print(f'      {OBJECT_DICT=}')
    OBJECT = klass("object", OBJECT_DICT)
    RANGE = klass("range", intersect(), implements=ITERABLE_INT)
    print(f'              {LEN=}')
    print(f'           {OBJECT=}')
    print(f'{ITERABLE_PROTOCOL=}')
    print(f'     {ITERABLE_INT=}')
    print(f'            {RANGE=}')
    print(f'           {OBJECT=}')
    
    
if __name__ == '__main__':
    pretty_print_type(parse_file('typeshed_mini/builtins.pyi'))
    # main()


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
