import numpy as np

def new(f): return f


@new
def list_of_ints() -> list[int]:
    return []


def empty(y: list[int]) -> list[int]:
    x = []
    # a = [x]
    for i in range(len(y)):
        x.append(i)
    return x


def build_list_of_lists(k: int) -> None:
    clusters = []
    for _ in range(k):
        clusters.append([])


def build_aliased_list_of_lists(k: int, xs: list[int], i: int) -> None:
    clusters = []
    for _ in range(k):
        clusters.append([])
    for x in xs:
        clusters[x].append(i)


def build_aliased_list_of_known_lists(k: int, xs: list[int], i: int) -> None:
    clusters = []
    for _ in range(k):
        clusters.append(list_of_ints())
    for x in xs:
        clusters[x].append(i)


def iterate(x: int):
    for i in range(x): pass


def cmp(a: float, b: float) -> bool:
    return a > b


def negative(b: bool) -> bool:
    return not b


def access(x: np.ndarray):
    y = x[0]
    z = x[1:]
    m = x[x]


def tup() -> tuple[int, float]:
    a, b = (1, 'x')

def listing():
    lst = [1]
    tpl = (1,)

def make_int() -> int:
    return 1

def empty_list_add(x: int):
    return [] + [x]

def list_append(y: int):
    x = []
    x.append(y)

def list_add():
    x = [1] + [make_int()]
    y = [(1,)] + [(2,)]

def pair() -> tuple[int, float]:
    x = 2
    res = x + 1
    return res

def destruct():
    a, b = pair()

def test_tuple(a: int, b: float) -> int:
    x = (a, b)
    (c, d) = x
    y = x[1]
    return y
