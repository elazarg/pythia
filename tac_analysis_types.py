# Data flow analysis and stuff.

from __future__ import annotations as _

from collections import defaultdict
from dataclasses import dataclass
from typing import TypeVar, TypeAlias, Optional

from frozendict import frozendict

import tac
from tac import Predefined
from tac_analysis_domain import Bottom, Lattice, BOTTOM

T = TypeVar('T')


@dataclass(frozen=True)
class ObjectType:
    name: str
    fields: frozendict[str, SimpleType]

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class Ref:
    name: str


@dataclass(frozen=True)
class FunctionType:
    return_type: SimpleType
    new: bool = True

    def __repr__(self):
        return f'() -> {self.return_type}'


@dataclass(frozen=True)
class Property:
    return_type: SimpleType
    new: bool = False

    def __repr__(self):
        return f'-> {self.return_type}'


SimpleType: TypeAlias = ObjectType | Ref | FunctionType


TypeElement: TypeAlias = ObjectType | FunctionType | Property | Ref | Bottom


def make_tuple(t: ObjectType) -> ObjectType:
    return ObjectType('tuple', frozendict({
        '__getitem__': FunctionType(t),
    }))


def iter_method(element_type: ObjectType) -> FunctionType:
    return FunctionType(ObjectType(f'iterator[{element_type}]', frozendict({
        '__next__': FunctionType(element_type),
    })))


OBJECT = ObjectType('object', frozendict({}))

FLOAT = ObjectType('float', frozendict({}))
INT = ObjectType('int', frozendict({}))
STRING = ObjectType('str', frozendict({}))
BOOL = ObjectType('bool', frozendict({}))
NONE = ObjectType('None', frozendict({}))
CODE = ObjectType('code', frozendict({}))
ASSERTION_ERROR = ObjectType('AssertionError', frozendict({}))

SLICE = ObjectType('slice', frozendict({}))

LIST = ObjectType('list', frozendict({
    '__getitem__': FunctionType(OBJECT, new=False),
    '__iter__': iter_method(OBJECT),
    '__len__': FunctionType(INT, new=False),
    '__contains__': FunctionType(BOOL, new=False),
    'clear': FunctionType(NONE, new=False),
    'copy': FunctionType(NONE, new=False),
    'count': FunctionType(INT, new=False),
    'extend': FunctionType(NONE),
    'index': FunctionType(INT, new=False),
    'insert': FunctionType(NONE, new=False),
    'pop': FunctionType(OBJECT),
    'remove': FunctionType(NONE, new=False),
    'reverse': FunctionType(NONE, new=False),
    'sort': FunctionType(NONE, new=False),
}))

TUPLE_CONSTRUCTOR = FunctionType(make_tuple(OBJECT), new=False)
LIST_CONSTRUCTOR = FunctionType(LIST, new=False)
SLICE_CONSTRUCTOR = FunctionType(SLICE, new=False)


NDARRAY = ObjectType('ndarray', frozendict({
    'mean': FunctionType(FLOAT, new=False),
    'std': FunctionType(FLOAT, new=False),
    'shape': make_tuple(INT),
    'size': INT,
    '__getitem__': FunctionType(FLOAT),
    '__iter__': iter_method(FLOAT),  # inaccurate
    'T': Property(Ref('numpy.ndarray'), new=True),
    'astype': FunctionType(Ref('numpy.ndarray'), new=True),
    'reshape': FunctionType(Ref('numpy.ndarray'), new=True),
    'ndim': INT,
}))

ARRAY_GEN = FunctionType(NDARRAY, new=True)

DATAFRAME = ObjectType('DataFrame', frozendict({}))

TIME_MODULE = ObjectType('/time', frozendict({}))

NUMPY_MODULE = ObjectType('/numpy', frozendict({
    'ndarray': NDARRAY,
    'array': ARRAY_GEN,
    'dot': FunctionType(FLOAT, new=False),
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
    'max': FunctionType(FLOAT, new=False),
    'min': FunctionType(FLOAT, new=False),
    'sum': FunctionType(FLOAT, new=False),
    'setdiff1d': ARRAY_GEN,
    'unique': ARRAY_GEN,
    'append': ARRAY_GEN,
    'random': ARRAY_GEN,
    'argmax': FunctionType(INT, new=False),
    'c_': ObjectType('slice_trick', {
        '__getitem__': FunctionType(NDARRAY),
    }),
    'r_': ObjectType('slice_trick', {
        '__getitem__': FunctionType(NDARRAY),
    }),

}))

SKLEARN_MODULE = ObjectType('/sklearn', frozendict({
    'metrics': ObjectType('/metrics', frozendict({
        'log_loss': FunctionType(FLOAT, new=False),
        'accuracy_score': FunctionType(FLOAT, new=False),
        'f1_score': FunctionType(FLOAT, new=False),
        'precision_score': FunctionType(FLOAT, new=False),
        'recall_score': FunctionType(FLOAT, new=False),
        'roc_auc_score': FunctionType(FLOAT, new=False),
        'average_precision_score': FunctionType(FLOAT, new=False),
        'roc_curve': FunctionType(make_tuple(FLOAT), new=False),
        'confusion_matrix': FunctionType(make_tuple(INT), new=False),
    })),
    'linear_model': ObjectType('/linear_model', frozendict({
        'LogisticRegression': FunctionType(ObjectType('LogisticRegression', frozendict({
            'fit': FunctionType(ObjectType('Model', {
                'predict': FunctionType(FLOAT),
                'predict_proba': ARRAY_GEN,
                'score': FunctionType(FLOAT),
            }), new=False),
        })), new=True),
        'LinearRegression': FunctionType(ObjectType('LinearRegression', frozendict({
            'fit': FunctionType(ObjectType('Model', {
                'predict': FunctionType(FLOAT),
                'predict_proba': ARRAY_GEN,
                'score': FunctionType(FLOAT),
            }), new=False),
        })), new=True),
    })),
}))

PANDAS_MODULE = ObjectType('/pandas', frozendict({
    'DataFrame': FunctionType(DATAFRAME),
}))

BUILTINS_MODULE = ObjectType('/builtins', frozendict({
    'range': FunctionType(ObjectType('range', frozendict({
        '__iter__': iter_method(INT),
    }))),
    'len': FunctionType(INT, new=False),
    'print': FunctionType(NONE, new=False),
    'abs': FunctionType(FLOAT, new=False),
    'round': FunctionType(FLOAT, new=False),
    'min': FunctionType(FLOAT, new=False),
    'max': FunctionType(FLOAT, new=False),
    'sum': FunctionType(FLOAT, new=False),
    'all': FunctionType(BOOL, new=False),
    'any': FunctionType(BOOL, new=False),
    'int': FunctionType(INT, new=False),
    'float': FunctionType(FLOAT, new=False),
    'str': FunctionType(STRING, new=False),
    'bool': FunctionType(BOOL, new=False),
    'code': FunctionType(CODE, new=False),
    'AssertionError': FunctionType(ASSERTION_ERROR, new=False),
}))

GLOBALS_OBJECT = ObjectType('globals()', frozendict({
    'numpy': NUMPY_MODULE,
    'np': NUMPY_MODULE,
    'pandas': PANDAS_MODULE,
    'time': TIME_MODULE,
    'sklearn': SKLEARN_MODULE,
    'sk': SKLEARN_MODULE,
    'mt': SKLEARN_MODULE.fields['metrics'],
    **BUILTINS_MODULE.fields
}))

BINARY = {
    (INT, INT): {
        '/': FunctionType(FLOAT, new=False),
        **{op: FunctionType(INT, new=False) for op in ['+', '-', '*', '**', '//', '%']},
        **{op: FunctionType(INT, new=False) for op in ['&', '|', '^', '<<', '>>']},
        **{op: FunctionType(BOOL, new=False) for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (INT, FLOAT): {
        **{op: FunctionType(FLOAT, new=False) for op in ['+', '-', '*', '/', '**', '//', '%']},
        **{op: FunctionType(BOOL, new=False) for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (FLOAT, INT): {
        **{op: FunctionType(FLOAT, new=False) for op in ['+', '-', '*', '/', '**', '//', '%']},
        **{op: FunctionType(BOOL, new=False) for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (FLOAT, FLOAT): {
        **{op: FunctionType(FLOAT, new=False) for op in ['+', '-', '*', '/', '**', '//', '%']},
        **{op: FunctionType(BOOL, new=False) for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (NDARRAY, NDARRAY): {op: FunctionType(NDARRAY, new=True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (NDARRAY, INT): {op: FunctionType(NDARRAY, new=True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (INT, NDARRAY): {op: FunctionType(NDARRAY, new=True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (NDARRAY, FLOAT): {op: FunctionType(NDARRAY, new=True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (FLOAT, NDARRAY): {op: FunctionType(NDARRAY, new=True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
}


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
            return result
        return ref

    def call(self, function: TypeElement, args: list[TypeElement]) -> TypeElement:
        if self.is_top(function):
            return self.top()
        if self.is_bottom(function) or any(self.is_bottom(arg) for arg in args):
            return self.bottom()
        if not isinstance(function, FunctionType):
            print(f'{function} is not a function')
            return self.bottom()
        return self.resolve(function.return_type)

    def _get_binary_op(self, left: TypeElement, right: TypeElement, op: str) -> FunctionType | Bottom:
        if isinstance(op, tac.Var):
            op = op.name
        if self.is_top(left) or self.is_top(right):
            return self.top()
        if self.is_bottom(left) or self.is_bottom(right):
            return self.bottom()
        category = BINARY.get((left, right), {})
        res = category.get(op, self.top())
        return self.resolve(res)

    def binary(self, left: TypeElement, right: TypeElement, op: str) -> TypeElement:
        function = self._get_binary_op(left, right, op)
        if not self.is_top(function) and not isinstance(function, Bottom):
            return self.call(function, [left, right])
        return self.top()

    def predefined(self, name: Predefined) -> Optional[TypeElement]:
        match name:
            case Predefined.LIST: return LIST_CONSTRUCTOR
            case Predefined.TUPLE: return TUPLE_CONSTRUCTOR
            case Predefined.GLOBALS: return GLOBALS_OBJECT
            case Predefined.NONLOCALS: return NONLOCALS_OBJECT
            case Predefined.LOCALS: return LOCALS_OBJECT
            case Predefined.SLICE: return SLICE_CONSTRUCTOR
        return None

    def const(self, value: object) -> TypeElement:
        return self.call(self.annotation(type(value).__name__), [])\
            if value is not None else NONE

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
            case FunctionType():
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

    def imported(self, modname: str) -> TypeElement:
        return GLOBALS_OBJECT.fields[modname]

    def is_subtype(self, left: TypeElement, right: TypeElement) -> bool:
        return self.join(left, right) == right

    def is_supertype(self, left: TypeElement, right: TypeElement) -> bool:
        return self.join(left, right) == left

    def is_allocation_function(self, function: TypeElement):
        return isinstance(function, FunctionType) and function.new

    def is_allocation_binary(self, left: T, right: T, op: str) -> bool:
        function = self._get_binary_op(left, right, op)
        return self.is_allocation_function(function)

    def is_allocation_attribute(self, var: TypeElement, attr: str) -> bool:
        p = self._attribute(var, attr)
        return isinstance(p, Property) and p.new


unseen = defaultdict(set)
