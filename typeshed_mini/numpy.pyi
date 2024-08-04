
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
    def __radd__(self: ndarray, other: float) -> ndarray: ...

    @new
    def __rsub__(self: ndarray, other: float) -> ndarray: ...

    @new
    def __rmul__(self: ndarray, other: float) -> ndarray: ...

    @new
    def __rtruediv__(self: ndarray, other: float) -> ndarray: ...

    @new
    def __gt__(self: ndarray, other) -> ndarray: ...

    @new
    def __lt__(self: ndarray, other) -> ndarray: ...

    @new
    def __getitem__(self: ndarray, key: slice) -> ndarray: ...
    @new
    def __getitem__[*Args](self: ndarray, key: tuple[*Args]) -> ndarray: ...
    @new
    def __getitem__(self: ndarray, key: ndarray) -> ndarray: ...
    @new
    def __getitem__(self: ndarray, key: list[int]) -> ndarray: ...

    def __getitem__(self: ndarray, key: int) -> float: ...

    @update(ndarray)
    def __setitem__(self: ndarray, key: int, value: float) -> None: ...
    @update(ndarray)
    def __setitem__[*Args](self: ndarray, key: tuple[*Args], value) -> None: ...
    #
    # @new
    # def __iter__(self: ndarray) -> Iterator[float]: ...

    @new
    def astype(self: ndarray, dtype) -> list[int]: ...

    @new
    def mean(self: ndarray) -> ndarray: ...
    def std(self: ndarray) -> float: ...

    @property
    @new
    def shape(self: ndarray) -> list[int]: ...

    def any(self: ndarray) -> bool: ...
    def all(self: ndarray) -> bool: ...

    @new
    def reshape(self: ndarray, shape: tuple) -> ndarray: ...
    @new
    def reshape(self: ndarray, shape: int) -> ndarray: ...
    @new
    def reshape(self: ndarray, d1: int, d2: int) -> ndarray: ...

# class c_:
#     def __getitem__[*Args](self: c_, key: slice | tuple[*Args]) -> ndarray: ...
@property
@new
def c_() -> ndarray: ...

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

@module
class random:
    @staticmethod
    @new
    def rand(dims: tuple) -> ndarray: ...

    @staticmethod
    def seed(seed: int) -> None: ...

    @staticmethod
    @new
    def choice(a: ndarray | int, size: int) -> ndarray: ...

@module
class linalg:
    @staticmethod
    @new
    def norm(a: ndarray, axis: int) -> ndarray: ...

@new
def array(object) -> ndarray: ...
@new
def array(object, n) -> ndarray: ...
