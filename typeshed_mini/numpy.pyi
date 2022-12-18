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

class ndarray:
    def size(self) -> int:...
    @new
    def __add__(self, other: ndarray) -> ndarray: ...
    @new
    def __sub__(self, other: ndarray) -> ndarray: ...
    @new
    def __mul__(self, other: ndarray) -> ndarray: ...
    def __gt__(self, other: ndarray) -> ndarray: ...
    def __getitem__(self, key: slice) -> float: ...

    @new
    def __iter__(self) -> SupportsNext[float]: ...
    @new
    def astype(self, dtype) -> ndarray: ...

@new
def setdiff1d(a: ndarray, b: ndarray) -> ndarray: ...
@new
def unique(arg: ndarray) -> ndarray: ...
@new
def append(arr: ndarray, values: ndarray) -> ndarray: ...

@new
def zeros(dims: tuple) -> ndarray: ...

def sum(x: ndarray) -> float: ...

class random:
    @new
    def rand(self, dims: tuple[int]) -> ndarray: ...

@new
def array(object, dtype) -> ndarray: ...
