# Data flow analysis and stuff.

from __future__ import annotations as _

from collections import defaultdict
from dataclasses import dataclass
from typing import TypeVar, TypeAlias, Optional, Iterable

from frozendict import frozendict

import tac
from tac import Predefined
from tac_analysis_domain import Bottom, Lattice, BOTTOM

T = TypeVar('T')


@dataclass(frozen=True)
class ObjectType:
    name: str
    fields: frozendict[str, SimpleType] = frozendict({})

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class Ref:
    name: str


@dataclass(frozen=True)
class ParamsType:
    params: tuple[SimpleType, ...]
    varargs: bool = False

    def __repr__(self):
        params = ", ".join(map(str, self.params))
        varargs = "*" if self.varargs else ""
        if params:
            varargs = ', ' + varargs
        return f'({params}{varargs})'


@dataclass(frozen=True)
class FunctionType:
    params: ParamsType
    return_type: SimpleType
    new: bool = True

    def __repr__(self):
        return f'{self.params} -> {"new " if self.new else ""}{self.return_type}'


@dataclass(frozen=True)
class OverloadedFunctionType:
    types: tuple[FunctionType, ...]

    def __repr__(self):
        return f'({", ".join(map(str, self.types))})'


def make_function_type(return_type: SimpleType, new: bool = True, pure=True) -> OverloadedFunctionType:
    return OverloadedFunctionType((FunctionType(ParamsType((), varargs=True), return_type, new=new),))


@dataclass(frozen=True)
class TypeType:
    type: SimpleType

    def __repr__(self):
        return f'Type[{self.type}]'


@dataclass(frozen=True)
class Property:
    return_type: SimpleType
    new: bool = False

    def __repr__(self):
        return f'-> {self.return_type}'


SimpleType: TypeAlias = ObjectType | Ref | FunctionType


TypeElement: TypeAlias = ObjectType | OverloadedFunctionType | Property | Ref | Bottom


def make_tuple(t: ObjectType) -> ObjectType:
    return ObjectType('tuple', frozendict({
        '__getitem__': make_function_type(t),
    }))


def iter_method(element_type: ObjectType) -> OverloadedFunctionType:
    return make_function_type(ObjectType(f'iterator[{element_type}]', frozendict({
        '__next__': make_function_type(element_type),
    })))


OBJECT = ObjectType('object')

FLOAT = ObjectType('float')
INT = ObjectType('int')
STRING = ObjectType('str')
BOOL = ObjectType('bool')
NONE = ObjectType('None')
CODE = ObjectType('code')
ASSERTION_ERROR = ObjectType('AssertionError')

SLICE = ObjectType('slice')

LIST = ObjectType('list', frozendict({
    '__getitem__': make_function_type(OBJECT, new=False),
    '__iter__': iter_method(OBJECT),
    '__len__': make_function_type(INT, new=False),
    '__contains__': make_function_type(BOOL, new=False),
    'clear': make_function_type(NONE, new=False),
    'copy': make_function_type(NONE, new=False),
    'count': make_function_type(INT, new=False),
    'extend': make_function_type(NONE),
    'index': make_function_type(INT, new=False),
    'insert': make_function_type(NONE, new=False),
    'pop': make_function_type(OBJECT),
    'remove': make_function_type(NONE, new=False),
    'reverse': make_function_type(NONE, new=False),
    'sort': make_function_type(NONE, new=False),
}))

DICT = ObjectType('dict', frozendict({
    '__getitem__': make_function_type(OBJECT, new=False),
    '__setitem__': make_function_type(NONE, new=False),
}))

TUPLE_CONSTRUCTOR = make_function_type(make_tuple(OBJECT), new=False)
LIST_CONSTRUCTOR = make_function_type(LIST, new=False)
SLICE_CONSTRUCTOR = make_function_type(SLICE, new=False)
DICT_CONSTRUCTOR = make_function_type(DICT, new=True)


NDARRAY = ObjectType('ndarray', frozendict({
    'mean': make_function_type(FLOAT, new=False, pure=True),
    'std': make_function_type(FLOAT, new=False, pure=True),
    'shape': make_tuple(INT),
    'size': INT,
    '__getitem__': OverloadedFunctionType(
        tuple([
            FunctionType(ParamsType((INT,), varargs=False), FLOAT, new=True),
            FunctionType(ParamsType((FLOAT,), varargs=False), FLOAT, new=True), # Fix: ad-hoc
            FunctionType(ParamsType((SLICE,), varargs=False), Ref('numpy.ndarray'), new=True),
            FunctionType(ParamsType((Ref('numpy.ndarray'),), varargs=False), Ref('numpy.ndarray'), new=True),
            FunctionType(ParamsType((make_tuple(OBJECT),), varargs=False), OBJECT, new=True),  # FIX: not necessarily new
        ])
    ),
    '__iter__': iter_method(FLOAT),  # inaccurate
    'T': Property(Ref('numpy.ndarray'), new=True),
    'astype': make_function_type(Ref('numpy.ndarray'), new=True, pure=True),
    'reshape': make_function_type(Ref('numpy.ndarray'), new=True, pure=False),
    'ndim': INT,
}))

ARRAY_GEN = make_function_type(NDARRAY, new=True)

DATAFRAME = ObjectType('DataFrame', frozendict({
    'loc': DICT,
}))

TIME_MODULE = ObjectType('/time')

NUMPY_MODULE = ObjectType('/numpy', frozendict({
    'ndarray': TypeType(NDARRAY),
    'array': ARRAY_GEN,
    'dot': make_function_type(FLOAT, new=False),
    'zeros': ARRAY_GEN,
    'ones': ARRAY_GEN,
    'concatenate': ARRAY_GEN,
    'empty': ARRAY_GEN,
    'empty_like': ARRAY_GEN,
    'full': ARRAY_GEN,
    'full_like': ARRAY_GEN,
    'arange': ARRAY_GEN,
    'linspace': ARRAY_GEN,
    'logspace': ARRAY_GEN,
    'geomspace': ARRAY_GEN,
    'meshgrid': ARRAY_GEN,
    'max': make_function_type(FLOAT, new=False),
    'min': make_function_type(FLOAT, new=False),
    'sum': make_function_type(FLOAT, new=False),
    'setdiff1d': ARRAY_GEN,
    'unique': ARRAY_GEN,
    'append': ARRAY_GEN,
    'random': ObjectType('/numpy.random', frozendict({
        'rand': ARRAY_GEN,
    })),
    'argmax': make_function_type(INT, new=False),
    'c_': ObjectType('slice_trick', frozendict({
        '__getitem__': make_function_type(NDARRAY),
    })),
    'r_': ObjectType('slice_trick', frozendict({
        '__getitem__': make_function_type(NDARRAY),
    })),
}))

SKLEARN_MODULE = ObjectType('/sklearn', frozendict({
    'metrics': ObjectType('/metrics', frozendict({
        'log_loss': make_function_type(FLOAT, new=False),
        'accuracy_score': make_function_type(FLOAT, new=False),
        'f1_score': make_function_type(FLOAT, new=False),
        'precision_score': make_function_type(FLOAT, new=False),
        'recall_score': make_function_type(FLOAT, new=False),
        'roc_auc_score': make_function_type(FLOAT, new=False),
        'average_precision_score': make_function_type(FLOAT, new=False),
        'roc_curve': make_function_type(make_tuple(FLOAT), new=False),
        'confusion_matrix': make_function_type(make_tuple(INT), new=False),
    })),
    'linear_model': ObjectType('/linear_model', frozendict({
        'LogisticRegression': make_function_type(ObjectType('LogisticRegression', frozendict({
            'fit': make_function_type(ObjectType('Model', frozendict({
                'predict': make_function_type(FLOAT),
                'predict_proba': ARRAY_GEN,
                'score': make_function_type(FLOAT),
            })), new=False),
        })), new=True),
        'LinearRegression': make_function_type(ObjectType('LinearRegression', frozendict({
            'fit': make_function_type(ObjectType('Model', frozendict({
                'predict': make_function_type(FLOAT),
                'predict_proba': ARRAY_GEN,
                'score': make_function_type(FLOAT),
            })), new=False),
        })), new=True),
    })),
}))

PANDAS_MODULE = ObjectType('/pandas', frozendict({
    'DataFrame': TypeType(DATAFRAME),
}))

BUILTINS_MODULE = ObjectType('/builtins', frozendict({
    'range': make_function_type(ObjectType('range', frozendict({
        '__iter__': iter_method(INT),
    }))),
    'len': make_function_type(INT, new=False),
    'not': make_function_type(BOOL, new=False),
    'print': make_function_type(NONE, new=False),
    'abs': make_function_type(FLOAT, new=False),
    'round': make_function_type(FLOAT, new=False),
    'min': make_function_type(FLOAT, new=False),
    'max': make_function_type(FLOAT, new=False),
    'sum': make_function_type(FLOAT, new=False),
    'all': make_function_type(BOOL, new=False),
    'any': make_function_type(BOOL, new=False),

    'int': TypeType(INT),
    'float': TypeType(FLOAT),
    'str': TypeType(STRING),
    'bool': TypeType(BOOL),
    'code': TypeType(CODE),
    'AssertionError': TypeType(ASSERTION_ERROR),

    'Linear_Regression': make_function_type(NDARRAY, new=True),
}))

FUTURE_MODULE = ObjectType('/future', frozendict({
    'annotations': ObjectType('_'),
}))

MATPLOTLIB_MODULE = ObjectType('/matplotlib', frozendict({
    'pyplot': ObjectType('pyplot', frozendict({
        'plot': make_function_type(NONE, new=False),
        'show': make_function_type(NONE, new=False),
    })),
}))

PERSIST_MODULE = ObjectType('/persist', frozendict({
    'range': make_function_type(ObjectType('range', frozendict({
        '__iter__': iter_method(INT),
    }))),
}))

GLOBALS_OBJECT = ObjectType('globals()', frozendict({
    '__future__': FUTURE_MODULE,
    'numpy': NUMPY_MODULE,
    'np': NUMPY_MODULE,
    'pandas': PANDAS_MODULE,
    'pd': PANDAS_MODULE,
    'time': TIME_MODULE,
    'sklearn': SKLEARN_MODULE,
    'sk': SKLEARN_MODULE,
    'pymm': ObjectType('/pymm'),
    'mt': SKLEARN_MODULE.fields['metrics'],
    'matplotlib': MATPLOTLIB_MODULE,
    'persist': PERSIST_MODULE,
    **BUILTINS_MODULE.fields
}))


def transpose(double_dispatch_table: dict[tuple[SimpleType, SimpleType], dict[str, tuple[SimpleType, bool]]])\
        -> dict[str, OverloadedFunctionType]:
    ops = ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']
    result: dict[str, list[FunctionType]] = {op: [] for op in ops}
    for op in ops:
        for params, table in double_dispatch_table.items():
            if op in table:
                return_type, new = table[op]
                result[op].append(FunctionType(ParamsType(params), return_type, new))
    return {op: OverloadedFunctionType(tuple(result[op])) for op in ops}


BINARY = transpose({
    (INT, INT): {
        '/':   (FLOAT, False),
        **{op: (INT, False) for op in ['+', '-', '*', '**', '//', '%']},
        **{op: (INT, False) for op in ['&', '|', '^', '<<', '>>']},
        **{op: (BOOL, False) for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (INT, FLOAT): {
        **{op: (FLOAT, False) for op in ['+', '-', '*', '/', '**', '//', '%']},
        **{op: (BOOL, False) for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (FLOAT, INT): {
        **{op: (FLOAT, False) for op in ['+', '-', '*', '/', '**', '//', '%']},
        **{op: (BOOL, False) for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (FLOAT, FLOAT): {
        **{op: (FLOAT, False) for op in ['+', '-', '*', '/', '**', '//', '%']},
        **{op: (BOOL, False) for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (NDARRAY, NDARRAY): {op: (NDARRAY, True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (NDARRAY, INT):     {op: (NDARRAY, True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (INT, NDARRAY):     {op: (NDARRAY, True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (NDARRAY, FLOAT):   {op: (NDARRAY, True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (FLOAT, NDARRAY):   {op: (NDARRAY, True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
})


class TypeLattice(Lattice[TypeElement]):
    """
    Abstract domain for type analysis with lattice operations.
    """

    def name(self) -> str:
        return "Type"

    def join(self, left: TypeElement, right: TypeElement) -> TypeElement:
        if self.is_bottom(left) or self.is_top(right):
            return right
        if self.is_bottom(right) or self.is_top(left):
            return left
        if left == right:
            return left
        return self.top()

    def meet(self, left: TypeElement, right: TypeElement) -> TypeElement:
        if self.is_top(left) or self.is_bottom(right):
            return left
        if self.is_top(right) or self.is_bottom(left):
            return right
        if left == right:
            return left
        return self.bottom()

    def top(self) -> TypeElement:
        return OBJECT

    def is_top(self, elem: TypeElement) -> bool:
        return elem == OBJECT

    def is_bottom(self, elem: TypeElement) -> bool:
        return elem == BOTTOM

    @classmethod
    def bottom(cls) -> TypeElement:
        return BOTTOM

    def resolve(self, ref: TypeElement) -> TypeElement:
        if isinstance(ref, Ref):
            result = GLOBALS_OBJECT
            for attr in ref.name.split('.'):
                result = result.fields[attr]
            assert isinstance(result, TypeType)
            return result.type
        return ref

    def is_match(self, args: list[TypeElement], params: ParamsType):
        if len(args) < len(params.params):
            return False
        if len(args) > len(params.params) and not params.varargs:
            return False
        return all(self.is_subtype(arg, param)
                   for arg, param in zip(args, params.params))

    def join_all(self, types: Iterable[TypeElement]) -> TypeElement:
        result = TypeLattice.bottom()
        for t in types:
            result = self.join(result, t)
        #
        # if self.is_bottom(result):
        #     return self.top()
        return result

    def narrow_overload(self, overload: OverloadedFunctionType, args: list[TypeElement]) -> OverloadedFunctionType:
        return OverloadedFunctionType(tuple(func for func in overload.types
                                            if self.is_match(args, func.params)))

    def call(self, function: TypeElement, args: list[TypeElement]) -> TypeElement:
        if self.is_top(function):
            return self.top()
        if self.is_bottom(function) or any(self.is_bottom(arg) for arg in args):
            return self.bottom()
        if isinstance(function, TypeType):
            return function.type
        if not isinstance(function, OverloadedFunctionType):
            print(f'{function} is not an overloaded function')
            return self.bottom()
        return self.join_all(self.resolve(func.return_type)
                             for func in self.narrow_overload(function, args).types)

    def _get_binary_op(self, op: str) -> OverloadedFunctionType:
        if isinstance(op, tac.Var):
            op = op.name
        return BINARY.get(op)

    def binary(self, left: TypeElement, right: TypeElement, op: str) -> TypeElement:
        overloaded_function = self._get_binary_op(op)
        return self.call(overloaded_function, [left, right])

    def predefined(self, name: Predefined) -> Optional[TypeElement]:
        match name:
            case Predefined.LIST: return LIST_CONSTRUCTOR
            case Predefined.TUPLE: return TUPLE_CONSTRUCTOR
            case Predefined.GLOBALS: return GLOBALS_OBJECT
            case Predefined.NONLOCALS: return NONLOCALS_OBJECT
            case Predefined.LOCALS: return LOCALS_OBJECT
            case Predefined.SLICE: return SLICE_CONSTRUCTOR
            case Predefined.CONST_KEY_MAP: return DICT_CONSTRUCTOR
        return None

    def const(self, value: object) -> TypeElement:
        if value is None:
            return NONE
        return self.annotation(type(value).__name__)

    def _attribute(self, var: TypeElement, attr: str) -> TypeElement:
        if self.is_top(var):
            return self.top()
        if self.is_bottom(var):
            return self.bottom()
        match var:
            case ObjectType() as obj:
                if attr in obj.fields:
                    return self.resolve(obj.fields[attr])
                else:
                    unseen[obj.name].add(attr)
                    return self.top()
            case TypeType():
                return self.bottom()
        return self.top()

    def attribute(self, var: TypeElement, attr: str) -> TypeElement:
        res = self._attribute(var, attr)
        if isinstance(res, Property):
            return res.return_type
        return res

    def subscr(self, array: TypeElement, index: TypeElement) -> TypeElement:
        if self.is_bottom(array):
            return self.bottom()
        if self.is_top(array):
            return self.top()
        f = array.fields.get('__getitem__')
        if f is None:
            print(f'{array} is not indexable')
            return self.bottom()
        return self.call(f, [index])

    def annotation(self, code: str) -> TypeElement:
        return self.resolve(Ref(code))
        return ObjectType(type(code).__name__, {})

    def imported(self, modname: tac.Var) -> TypeElement:
        if isinstance(modname, tac.Var):
            return GLOBALS_OBJECT.fields[modname.name]
        elif isinstance(modname, tac.Attribute):
            return self.attribute(modname.var, modname.field)

    def is_subtype(self, left: TypeElement, right: TypeElement) -> bool:
        left = self.resolve(left)
        right = self.resolve(right)
        return self.join(left, right) == right

    def is_supertype(self, left: TypeElement, right: TypeElement) -> bool:
        return self.join(left, right) == left

    def is_allocation_function(self, overloaded_function: TypeElement) -> tac.AllocationType:
        if isinstance(overloaded_function, OverloadedFunctionType):
            if all(function.new for function in overloaded_function.types):
                return tac.AllocationType.STACK
            if not any(function.new for function in overloaded_function.types):
                return tac.AllocationType.NONE
        return tac.AllocationType.UNKNOWN

    def is_allocation_binary(self, left: TypeElement, right: TypeElement, op: str) -> tac.AllocationType:
        overloaded_function = self._get_binary_op(op)
        narrowed = self.narrow_overload(overloaded_function, [left, right])
        if self.is_allocation_function(narrowed):
            return tac.AllocationType.STACK
        return tac.AllocationType.NONE

    def is_allocation_attribute(self, var: TypeElement, attr: str) -> tac.AllocationType:
        p = self._attribute(var, attr)
        if isinstance(p, Property) and p.new:
            return tac.AllocationType.STACK
        return tac.AllocationType.NONE


unseen = defaultdict(set)
