from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Optional, TypeAlias


class TypeExpr:
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


Field = typing.TypeVar('Field', str, int)


@dataclass(frozen=True)
class VarDecl(typing.Generic[Field]):
    name: Field
    type: TypeExpr

    def __repr__(self):
        return f'{self.name}: {self.type}'


@dataclass(frozen=True)
class MapType(typing.Generic[Field]):
    items: tuple[VarDecl, ...]

    def __repr__(self) -> str:
        return f'{{{", ".join(f"{decl.name}: {decl.type}" for decl in self.items)}}}'


Tuple: typing.TypeAlias = MapType[int]
Struct: typing.TypeAlias = MapType[str]


T = typing.TypeVar('T', bound=TypeExpr)


@dataclass(frozen=True)
class Generic(TypeExpr, typing.Generic[T]):
    type_params: tuple[TypeVar]
    type: T

    def __repr__(self) -> str:
        return f'[{", ".join(repr(x) for x in self.type_params)}]{self.type}'

    def __getitem__(self, *items: TypeExpr) -> Instantiation:
        return Instantiation(self, items)


@dataclass(frozen=True)
class Instantiation(TypeExpr):
    generic: Generic
    type_params: tuple[TypeExpr]

    def __repr__(self):
        return f'{self.generic}[{", ".join(repr(x) for x in self.type_params)}]'


@dataclass(frozen=True)
class FunctionType(TypeExpr):
    params: Optional[tuple[VarDecl, ...]]
    return_type: TypeExpr
    new: bool

    def __repr__(self) -> str:
        if self.params is None:
            return f'->{self.return_type}'
        else:
            params = ', '.join(repr(x) for x in self.params)
            return f'({params}) -> {"new " if self.new else ""}{self.return_type}'

    def is_property(self) -> bool:
        return self.params is None


@dataclass(frozen=True)
class OverloadedFunctionType:
    types: tuple[Generic[FunctionType], ...]

    def __repr__(self):
        return f'({", ".join(map(str, self.types))})'


def make_function_type(return_type: TypeExpr,
                       params: tuple[VarDecl, ...] = (),
                       new: bool = True,
                       pure=True) -> OverloadedFunctionType:
    f = FunctionType(params, return_type, new=new)
    g = Generic((), f)
    return OverloadedFunctionType((g,))


SimpleType: TypeAlias = Ref | TypeVar | OverloadedFunctionType | MapType | Generic | Instantiation | FunctionType


def is_generic(self):
    return all(v is None for v in self.type_params.values())


def simplify_generic(t, context: dict[TypeVar, TypeExpr]):
    match t:
        case Generic(type_params, type):
            new_context = {k: v for k, v in context.items() if k not in type_params}
            return simplify_generic(type, new_context)
        case Instantiation(generic, type_params):
            new_type_params = tuple(simplify_generic(t, context) for t in type_params)
            new_context = {**context, **dict(zip(generic.type_params, new_type_params))}
            return simplify_generic(generic.type, new_context)
        case Ref():
            return t
        case TypeVar():
            return context.get(t, t)
        case MapType(items):
            return MapType(tuple(VarDecl(item.name, simplify_generic(item.type, context))
                                 for item in items))
        case FunctionType(params, return_type, new):
            new_params = tuple(VarDecl(item.name, simplify_generic(item.type, context))
                               for item in params)
            new_return_type = simplify_generic(return_type, context)
            print(return_type, new_return_type)
            return FunctionType(new_params, new_return_type, new)
        case OverloadedFunctionType(types): raise NotImplementedError
        case _: raise NotImplementedError(f'{t!r}')


def main():
    LEN = Generic((TypeVar('T'),), FunctionType((VarDecl('obj', TypeVar('T')),), TypeVar('int'), new=False))
    print(LEN)

    NEXT_METHOD = VarDecl('__next__', FunctionType((VarDecl('self', TypeVar('Self')),), TypeVar('T'), new=False))

    ITERATOR_PROTOCOL = Struct((NEXT_METHOD,))

    ITERABLE_METHOD = FunctionType((VarDecl('self', TypeVar('Self')),), ITERATOR_PROTOCOL, new=False)

    ITERABLE_PROTOCOL = Generic((TypeVar('T'),), Struct(
        (VarDecl('__iter__', ITERABLE_METHOD),)
    ))

    print(ITERABLE_PROTOCOL[TypeVar('int')])
    print(simplify_generic(ITERABLE_PROTOCOL[TypeVar('int')], {}))


if __name__ == '__main__':
    main()

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
