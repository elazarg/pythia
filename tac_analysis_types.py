# Data flow analysis and stuff.

from __future__ import annotations as _

import ast
from collections import defaultdict
from dataclasses import dataclass
from typing import TypeAlias, Optional, Iterable
import typing

from frozendict import frozendict

import tac
from tac import Predefined, UnOp
from tac_analysis_domain import Bottom, Lattice, BOTTOM


@dataclass(frozen=True)
class Ref:
    name: str
    static: bool = True


@dataclass(frozen=True)
class TypeVar:
    name: str


@dataclass(frozen=True)
class ObjectType:
    name: str
    fields: frozendict[str, SimpleType] = frozendict({})
    type_params: frozendict[TypeVar, Optional[SimpleType]] = frozendict({})

    def __repr__(self):
        if self.type_params:
            return f'{self.name}[{", ".join(f"{v}" for k, v in self.type_params.items())}]'
        else:
            return self.name

    def __getitem__(self, *items):
        return instantiate(self, {k: item for k, item in zip(self.type_params.keys(), items)})


@dataclass(frozen=True)
class FunctionType:
    params: ParamsType
    return_type: SimpleType
    new: bool = True
    type_params: frozendict[TypeVar, Optional[SimpleType]] = frozendict({})

    def __repr__(self):
        return f'{self.params} -> {"new " if self.new else ""}{self.return_type}'


@dataclass(frozen=True)
class ParamsType:
    params: tuple[SimpleType | tac.Var, ...]
    varargs: bool = False

    def __repr__(self):
        params = ", ".join(map(str, self.params))
        varargs = "*" if self.varargs else ""
        if params:
            varargs = ', ' + varargs
        return f'({params}{varargs})'


@dataclass(frozen=True)
class OverloadedFunctionType:
    types: tuple[FunctionType, ...]

    def __repr__(self):
        return f'({", ".join(map(str, self.types))})'


def make_function_type(return_type: SimpleType, new: bool = True, pure=True) -> OverloadedFunctionType:
    return OverloadedFunctionType((FunctionType(ParamsType((), varargs=True), return_type, new=new),))


@dataclass(frozen=True)
class Property:
    return_type: SimpleType
    new: bool = False

    def __repr__(self):
        return f'-> {self.return_type}'


SimpleType: TypeAlias = ObjectType | Ref | FunctionType | OverloadedFunctionType | TypeVar


TypeElement: TypeAlias = ObjectType | OverloadedFunctionType | Property | Ref | Bottom


def is_generic(self):
    return all(v is None for v in self.type_params.values())


T = typing.TypeVar('T')


def instantiate(self: T, type_args: dict[TypeVar, SimpleType]) -> T:
    assert is_generic(self)
    match self:
        case ObjectType(name, fields, type_params):
            assert list(type_args.keys()) == list(self.type_params.keys()), f'{type_args} {self.type_params}'
            type_params = {k: type_args[k] for k in type_params}
            fields = {k: bind(v, type_params) for k, v in fields.items()}
            return ObjectType(name, frozendict(fields), frozendict(type_params))
        case OverloadedFunctionType(overloads):
            return OverloadedFunctionType(tuple(instantiate(overload, type_args)
                                                for overload in overloads))
        case FunctionType(params, return_type, new, type_params):
            assert list(type_args.keys()) == list(self.type_params.keys()), f'{type_args} {self.type_params}'
            type_params = {k: v for k, v in zip(type_params.keys(), type_args)}
            return FunctionType(params=bind(params, type_args),
                                return_type=bind(return_type, type_args),
                                new=new,
                                type_params=frozendict(type_params))
        case _: raise NotImplementedError(f'{self}')


def bind(self: T, type_args: dict[TypeVar, SimpleType]) -> T:
    match self:
        case ObjectType(name, fields, type_params):
            type_args = {k: v for k, v in type_args.items() if k not in type_params}
            type_params = {k: (bind(v, type_args) if v is not None else None)
                           for k, v in type_params.items()}
            fields = {k: bind(v, type_args) for k, v in fields.items()}
            return ObjectType(name, frozendict(fields), frozendict(type_params))
        case FunctionType(params, return_type, new, type_params):
            return FunctionType(params=bind(params, type_args),
                                return_type=bind(return_type, type_args),
                                new=new,
                                type_params=type_params)
        case OverloadedFunctionType(overloads):
            return OverloadedFunctionType(tuple(bind(overload, type_args)
                                                for overload in overloads))
        case ParamsType(params, varargs):
            return ParamsType(tuple(instantiate(t, type_args) if isinstance(t, SimpleType) else type_args[t]
                                    for t in params),
                              varargs)
        case TypeVar(name):
            return type_args.get(self, self)
        case _: raise NotImplementedError(f'{self}')


def make_tuple(t: ObjectType) -> ObjectType:
    return ObjectType('tuple', frozendict({
        '__getitem__': make_function_type(t),
    }))


def iter_method(element_type: ObjectType) -> OverloadedFunctionType:
    return make_function_type(ObjectType(f'iterator', frozendict({
        '__next__': make_function_type(element_type),
    }), type_params=frozendict({'Iterator': element_type})), new=True)


OBJECT = ObjectType('object')

FLOAT = ObjectType('float')
INT = ObjectType('int')
STRING = ObjectType('str')
BOOL = ObjectType('bool', frozendict({
    '__bool__': make_function_type(Ref('bool'), new=False),
}))

TYPE = ObjectType('type', frozendict({
    '__call__': make_function_type(TypeVar('T'), new=False),
}), type_params=frozendict({'T': None}))


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
}), type_params=frozendict({'T': None}))

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
            FunctionType(ParamsType((FLOAT,), varargs=False), FLOAT, new=True),  # Fix: ad-hoc
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
    'ndarray': TYPE[NDARRAY],
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
    'DataFrame': TYPE[DATAFRAME],
}))

BUILTINS_MODULE = ObjectType('/builtins', frozendict({
    'range': make_function_type(ObjectType('range', frozendict({
        '__iter__': iter_method(INT),
    }))),
    'zip': make_function_type(ObjectType('zip', frozendict({
        '__iter__': iter_method(make_tuple(OBJECT)),
    }))),

    'len': make_function_type(INT, new=False),
    'print': make_function_type(NONE, new=False),
    'abs': make_function_type(FLOAT, new=False),
    'round': make_function_type(FLOAT, new=False),
    'min': make_function_type(FLOAT, new=False),
    'max': make_function_type(FLOAT, new=False),
    'sum': make_function_type(FLOAT, new=False),
    'all': make_function_type(BOOL, new=False),
    'any': make_function_type(BOOL, new=False),

    'int': TYPE[INT],
    'float': TYPE[FLOAT],
    'str': TYPE[STRING],
    'bool': TYPE[BOOL],
    'code': TYPE[CODE],
    'object': TYPE[OBJECT],
    'AssertionError': TYPE[ASSERTION_ERROR],
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

MODULES = frozendict({
    '__future__': FUTURE_MODULE,
    'numpy': NUMPY_MODULE,
    'pandas': PANDAS_MODULE,
    'time': TIME_MODULE,
    'sklearn': SKLEARN_MODULE,
    'matplotlib': MATPLOTLIB_MODULE,
    'persist': PERSIST_MODULE,
    'builtins': BUILTINS_MODULE,
})


def resolve_module(path: str, modules: dict = MODULES) -> ObjectType:
    path = path.split('.')
    obj = modules[path[0]]
    for part in path[1:]:
        obj = obj.fields[part]
    return obj


def transpose(double_dispatch_table: dict[tuple[SimpleType, SimpleType], dict[str, tuple[SimpleType, bool]]]
              ) -> dict[str, OverloadedFunctionType]:
    ops = ['+', '-', '*', '/', '**', '//', '%', '@',
           '==', '!=', '<', '<=', '>', '>=']
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
        **{op: (INT, False) for op in ['+', '-', '*',       '**', '//', '%']},
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


def parse_type_annotation(expr: ast.expr, global_state: dict[str, SimpleType]) -> SimpleType:
    print(expr)
    match expr:
        case ast.Name(id=name):
            res = global_state[name]
            assert res.name == 'type'
            return res.fields['T']
        case ast.Attribute(value=ast.Name(id=name), attr=attr):
            obj = global_state[name]
            res = obj.fields[attr]
            assert res.name == 'type'
            return res.type_params['T']
        case ast.Subscript(value=ast.Name(id=generic), slice=index):
            if generic == 'tuple':
                return make_tuple(OBJECT)
            generic = global_state[generic]
            index = global_state[index]
            return generic[index]
        case ast.Constant(value=None):
            return NONE
    assert False, f'Unsupported type annotation: {expr}'


class TypeLattice(Lattice[TypeElement]):
    """
    Abstract domain for type analysis with lattice operations.
    """
    def __init__(self, functions, imports: dict[str, str]):
        global_state = {
            **{name: resolve_module(path) for name, path in imports.items()},
            **BUILTINS_MODULE.fields,
        }

        global_state.update({f: make_function_type(
                                    return_type=parse_type_annotation(
                                        ast.parse(functions[f].__annotations__.get('return', 'object'), mode='eval').body,
                                        global_state,
                                    ),
                                    new=False,
                                )
                             for f in functions})

        self.globals = ObjectType('globals()', frozendict(global_state))

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
            if ref.static:
                result = resolve_module(ref.name)
                assert result.name == 'type'
                return result.type_params['T']
            result = self.globals
            for attr in ref.name.split('.'):
                result = result.fields[attr]
            assert result.name == 'type'
            return result.type_params['T']
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
        return result

    def narrow_overload(self, overload: OverloadedFunctionType, args: list[TypeElement]) -> OverloadedFunctionType:
        return OverloadedFunctionType(tuple(func for func in overload.types
                                            if self.is_match(args, func.params)))

    def call(self, function: TypeElement, args: list[TypeElement]) -> TypeElement:
        if self.is_top(function):
            return self.top()
        if self.is_bottom(function) or any(self.is_bottom(arg) for arg in args):
            return self.bottom()
        if isinstance(function, ObjectType):
            if function.fields.get('__call__'):
                return self.call(function.fields['__call__'], args)
        if not isinstance(function, OverloadedFunctionType):
            print(f'{function} is not an overloaded function')
            return self.bottom()
        return self.join_all(self.resolve(func.return_type)
                             for func in self.narrow_overload(function, args).types)

    def binary(self, left: TypeElement, right: TypeElement, op: str) -> TypeElement:
        overloaded_function = BINARY.get(op)
        result = self.call(overloaded_function, [left, right])
        # Shorthand for: "we assume that there is an implementation"
        if self.is_bottom(result):
            return self.top()
        return result

    def get_unary_attribute(self, value: TypeElement, op: UnOp) -> TypeElement:
        match op:
            case UnOp.NOT: dunder = '__bool__'
            case UnOp.POS: dunder = '__pos__'
            case UnOp.INVERT: dunder = '__invert__'
            case UnOp.NEG: dunder = '__neg__'
            case UnOp.ITER | UnOp.YIELD_ITER: dunder = '__iter__'
            case UnOp.NEXT: dunder = '__next__'
            case x:
                raise NotImplementedError(f'Unary operation {x} not implemented')
        return self._attribute(value, tac.Var(dunder))

    def unary(self, value: TypeElement, op: UnOp) -> TypeElement:
        if op == UnOp.NOT:
            return BOOL
        f = self.get_unary_attribute(value, op)
        return self.call(f, [value])

    def predefined(self, name: Predefined) -> Optional[TypeElement]:
        match name:
            case Predefined.LIST: return LIST_CONSTRUCTOR
            case Predefined.TUPLE: return TUPLE_CONSTRUCTOR
            case Predefined.GLOBALS: return self.globals
            case Predefined.NONLOCALS: return NONLOCALS_OBJECT
            case Predefined.LOCALS: return LOCALS_OBJECT
            case Predefined.SLICE: return SLICE_CONSTRUCTOR
            case Predefined.CONST_KEY_MAP: return DICT_CONSTRUCTOR
        return None

    def const(self, value: object) -> TypeElement:
        if value is None:
            return NONE
        return self.annotation(type(value).__name__)

    def _attribute(self, var: TypeElement, attr: tac.Var) -> TypeElement:
        if self.is_top(var):
            return self.top()
        if self.is_bottom(var):
            return self.bottom()
        match var:
            case ObjectType() as obj:
                name = attr.name
                if name in obj.fields:
                    return self.resolve(obj.fields[name])
                else:
                    unseen[obj.name].add(name)
                    return self.top()
        return self.top()

    def attribute(self, var: TypeElement, attr: tac.Var) -> TypeElement:
        assert isinstance(attr, tac.Var)
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
            print(f'{array} is not subscriptable')
            return self.bottom()
        return self.call(f, [index])

    def annotation(self, code: str) -> TypeElement:
        return self.resolve(Ref(code, static=False))
        return ObjectType(type(code).__name__, {})

    def imported(self, modname: tac.Var) -> TypeElement:
        if isinstance(modname, tac.Var):
            return resolve_module(modname.name)
        elif isinstance(modname, tac.Attribute):
            return self.attribute(modname.var, modname.field)

    def is_subtype(self, left: TypeElement, right: TypeElement) -> bool:
        left = self.resolve(left)
        right = self.resolve(right)
        return self.join(left, right) == right

    def is_supertype(self, left: TypeElement, right: TypeElement) -> bool:
        return self.join(left, right) == left

    def allocation_type_function(self, overloaded_function: TypeElement) -> tac.AllocationType:
        if isinstance(overloaded_function, OverloadedFunctionType):
            if all(function.new for function in overloaded_function.types):
                return tac.AllocationType.STACK
            if not any(function.new for function in overloaded_function.types):
                return tac.AllocationType.NONE
        return tac.AllocationType.UNKNOWN

    def allocation_type_binary(self, left: TypeElement, right: TypeElement, op: str) -> tac.AllocationType:
        overloaded_function = BINARY.get(op)
        narrowed = self.narrow_overload(overloaded_function, [left, right])
        if self.allocation_type_function(narrowed):
            return tac.AllocationType.STACK
        return tac.AllocationType.NONE

    def allocation_type_unary(self, value: TypeElement, op: tac.UnOp) -> tac.AllocationType:
        if op is UnOp.NOT:
            return tac.AllocationType.NONE
        f = self.get_unary_attribute(value, op)
        if f.types[0].new:
            return tac.AllocationType.STACK
        return tac.AllocationType.NONE

    def allocation_type_attribute(self, var: TypeElement, attr: tac.Var) -> tac.AllocationType:
        p = self._attribute(var, attr)
        if isinstance(p, Property) and p.new:
            return tac.AllocationType.STACK
        return tac.AllocationType.NONE


unseen = defaultdict(set)
