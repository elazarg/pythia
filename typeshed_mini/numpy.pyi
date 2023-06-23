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
T = TypeVar('T')
Q = TypeVar('Q')
Args = TypeVarTuple('Args')
N = TypeVar('N', Literal[int])

class ndarray:
    @property
    def size(self: ndarray) -> int:...
    @property
    def ndim(self: ndarray) -> int:...
    @property
    @new
    def T(self: ndarray) -> ndarray:...
    @new
    def __add__(self: ndarray, other: ndarray) -> ndarray: ...

    @new
    def __sub__(self: ndarray, other: ndarray) -> ndarray: ...

    @new
    def __mul__(self: ndarray, other: ndarray) -> ndarray: ...

    @new
    def __truediv__(self: ndarray, other: ndarray) -> ndarray: ...
    @new
    def __truediv__(self: ndarray, other: float) -> ndarray: ...

    @new
    def __radd__(self, other: float) -> ndarray: ...

    @new
    def __rsub__(self, other: float) -> ndarray: ...

    @new
    def __rmul__(self, other: float) -> ndarray: ...

    @new
    def __rtruediv__(self, other: float) -> ndarray: ...

    @new
    def __gt__(self: ndarray, other) -> ndarray: ...

    @new
    def __lt__(self: ndarray, other) -> ndarray: ...
    @new
    def __getitem__(self: ndarray, key: slice) -> ndarray: ...
    @new
    def __getitem__(self: ndarray, key: tuple[*Args]) -> ndarray: ...
    @new
    def __getitem__(self: ndarray, key: ndarray) -> ndarray: ...
    @new
    def __getitem__(self: ndarray, key: list[int]) -> ndarray: ...

    def __getitem__(self: ndarray, key: int) -> float: ...

    @new
    def __iter__(self: ndarray) -> SupportsNext[float]: ...

    @new
    def astype(self: ndarray, dtype) -> ndarray: ...

    @new
    def mean(self: ndarray) -> ndarray: ...
    def std(self: ndarray) -> float: ...

    @property
    def shape(self: ndarray) -> list[int]: ...
    def any(self: ndarray) -> bool: ...
    def all(self: ndarray) -> bool: ...

    @new
    def reshape(self: ndarray, shape: tuple) -> ndarray: ...
    @new
    def reshape(self: ndarray, shape: int) -> ndarray: ...
    @new
    def reshape(self: ndarray, d1: int, d2: int) -> ndarray: ...

class c_:
    def __getitem__(self: c_, key: slice | tuple[*Args]) -> ndarray: ...

@new
def setdiff1d(a: ndarray, b: ndarray) -> ndarray: ...

@new
def unique(arg: ndarray) -> ndarray: ...

@new
def append(arr: ndarray, value: float) -> ndarray: ...
# def append(arr: ndarray, values: ndarray) -> ndarray: ...

@new
def zeros(dims: tuple) -> ndarray: ...

@new
def zeros(dims: int) -> ndarray: ...

@new
def ones(dims: tuple | int) -> ndarray: ...

@new
def mean(x: ndarray, axis: int) -> ndarray: ...

@new
def dot(x: ndarray, y: ndarray) -> ndarray: ...

def sum(x: ndarray) -> float: ...
def argmin(x: ndarray) -> int: ...

@new
def concatenate(arrays: tuple | ndarray) -> ndarray: ...


class random:
    @new
    def rand(self, dims: tuple) -> ndarray: ...

    @new
    def choice(self, a: ndarray, size: int) -> ndarray: ...
    @new
    def choice(self, a: int, size: int) -> ndarray: ...

class linalg:
    def norm(self, a: ndarray, axis: int) -> ndarray: ...
@new
def array(object) -> ndarray: ...
@new
def array(object, n) -> ndarray: ...
