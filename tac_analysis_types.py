# Data flow analysis and stuff.

from __future__ import annotations as _

from collections import defaultdict
from dataclasses import dataclass
from typing import TypeVar, TypeAlias, Optional

from tac import Predefined
from tac_analysis_domain import Bottom, Top, Lattice, BOTTOM

T = TypeVar('T')


@dataclass()
class ObjectType:
    name: str
    fields: dict[str, FunctionType | ObjectType]

    def __repr__(self):
        return self.name


@dataclass
class FunctionType:
    return_type: ObjectType
    new: bool = True

    def __repr__(self):
        return f'() -> {self.return_type}'


def make_tuple(t: ObjectType) -> ObjectType:
    return ObjectType('tuple', {
        '__getitem__': FunctionType(t),
    })


def iter_method(element_type: ObjectType) -> FunctionType:
    return FunctionType(ObjectType(f'iterator[{element_type}]', {
        '__next__': FunctionType(element_type),
    }))


OBJECT = ObjectType('object', {})

FLOAT = ObjectType('float', {})
INT = ObjectType('int', {})
STRING = ObjectType('str', {})
BOOL = ObjectType('bool', {})
NONE = ObjectType('None', {})

LIST = ObjectType('list', {
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
})

TUPLE_CONSTRUCTOR = FunctionType(make_tuple(OBJECT), new=False)
LIST_CONSTRUCTOR = FunctionType(LIST, new=False)

NDARRAY = ObjectType('ndarray', {
    'mean': FunctionType(FLOAT, new=False),
    'std': FunctionType(FLOAT, new=False),
    'shape': make_tuple(INT),
    'size': INT,
    '__getitem__': FunctionType(FLOAT),
    '__iter__': iter_method(FLOAT),  # inaccurate
})


NDARRAY.fields['T'] = NDARRAY

ARRAY_GEN = FunctionType(NDARRAY)

NDARRAY.fields['astype'] = ARRAY_GEN


DATAFRAME = ObjectType('DataFrame', {

})

TIME_MODULE = ObjectType('/time', {

})

NUMPY_MODULE = ObjectType('/numpy', {
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

})

PANDAS_MODULE = ObjectType('/pandas', {
    'DataFrame': FunctionType(DATAFRAME),
})

BUILTINS_MODULE = ObjectType('/builtins', {
    'range': FunctionType(ObjectType('range', {
        '__iter__': iter_method(INT),
    })),
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
})

GLOBALS_OBJECT = ObjectType('globals()', {
    'numpy': NUMPY_MODULE,
    'np': NUMPY_MODULE,
    'pandas': PANDAS_MODULE,
    'time': TIME_MODULE,
})
GLOBALS_OBJECT.fields.update(BUILTINS_MODULE.fields)

BINARY = {
    (INT.name, INT.name): {
        '/': FunctionType(FLOAT, new=False),
        **{op: FunctionType(INT, new=False) for op in ['+', '-', '*', '**', '//', '%']},
        **{op: FunctionType(INT, new=False) for op in ['&', '|', '^', '<<', '>>']},
        **{op: FunctionType(BOOL, new=False) for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (INT.name, FLOAT.name): {
        **{op: FunctionType(FLOAT, new=False) for op in ['+', '-', '*', '/', '**', '//', '%']},
        **{op: FunctionType(BOOL, new=False) for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (FLOAT.name, INT.name): {
        **{op: FunctionType(FLOAT, new=False) for op in ['+', '-', '*', '/', '**', '//', '%']},
        **{op: FunctionType(BOOL, new=False) for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (FLOAT.name, FLOAT.name): {
        **{op: FunctionType(FLOAT, new=False) for op in ['+', '-', '*', '/', '**', '//', '%']},
        **{op: FunctionType(BOOL, new=False) for op in ['<', '>', '<=', '>=', '==', '!=']},
    },
    (NDARRAY.name, NDARRAY.name): {op: FunctionType(NDARRAY, new=True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (NDARRAY.name, INT.name): {op: FunctionType(NDARRAY, new=True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (INT.name, NDARRAY.name): {op: FunctionType(NDARRAY, new=True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (NDARRAY.name, FLOAT.name): {op: FunctionType(NDARRAY, new=True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
    (FLOAT.name, NDARRAY.name): {op: FunctionType(NDARRAY, new=True) for op in ['+', '-', '*', '/', '**', '//', '%', '@', '==', '!=', '<', '<=', '>', '>=']},
}


TypeElement: TypeAlias = ObjectType | FunctionType | Top | Bottom


@dataclass(frozen=True)
class TypeLattice(Lattice[TypeElement]):
    """
    Abstract domain for type analysis with lattice operations.
    """

    @staticmethod
    def name() -> str:
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

    def call(self, function: TypeElement, args: list[TypeElement]) -> TypeElement:
        if self.is_top(function):
            return self.top()
        if self.is_bottom(function) or any(self.is_bottom(arg) for arg in args):
            return self.bottom()
        if not isinstance(function, FunctionType):
            print(f'call({function}, {args}) == {function} which is not a function')
            return self.bottom()
        # if function.name() == 'LIST':
        #     return LIST
        return function.return_type

    def binary(self, left: TypeElement, right: TypeElement, op: str) -> TypeElement:
        if self.is_top(left) or self.is_top(right):
            return self.top()
        if self.is_bottom(left) or self.is_bottom(right):
            return self.bottom()
        if (left.name, right.name) in BINARY:
            ftype = BINARY[(left.name, right.name)][op]
            return self.call(ftype, [left, right])
        return self.top()

    def predefined(self, name: Predefined) -> Optional[TypeElement]:
        match name:
            case Predefined.LIST: return LIST_CONSTRUCTOR
            case Predefined.TUPLE: return TUPLE_CONSTRUCTOR
            case Predefined.GLOBALS: return GLOBALS_OBJECT
        return None

    def const(self, value: object) -> TypeElement:
        return self.annotation(type(value).__name__) if value is not None else NONE

    def attribute(self, var: TypeElement, attr: str) -> TypeElement:
        if self.is_top(var):
            return self.top()
        if self.is_bottom(var):
            return self.bottom()
        match var:
            case ObjectType() as obj:
                if attr in obj.fields:
                    return obj.fields[attr]
                else:
                    unseen[obj.name].add(attr)
                return self.top()
            case FunctionType():
                return self.bottom()
        return self.top()

    def subscr(self, array: TypeElement, index: TypeElement) -> TypeElement:
        if self.is_bottom(array):
            return self.bottom()
        if self.is_top(array):
            return self.top()
        f = array.fields.get('__getitem__')
        if f is None:
            print(f'{array} is not an indexable type')
            return self.bottom()
        return self.call(f, [index])

    def annotation(self, code: str) -> TypeElement:
        match code:
            case "np.ndarray" | "numpy.ndarray" | "ndarray": return NDARRAY
            case "int": return INT
            case "float": return FLOAT
            case "str": return STRING
            case "bool": return BOOL
        return ObjectType(type(code).__name__, {})

    def imported(self, modname: str) -> TypeElement:
        return GLOBALS_OBJECT.fields[modname]


unseen = defaultdict(set)
