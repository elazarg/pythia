def empty_list_add(x: int) -> list[int]:
    return [] + [x]


def list_append(y: int):
    x = []
    x.append(y)


def list_set(x: list[int]):
    x[0] = ""


def list_add(k: int):
    x = [1] + [k]
    y = [(1,)] + [(2,)]
    return x, y


def append(y: list[int], xs: list[int]) -> list[int]:
    x = []
    # a = [x]
    for i in xs:  # type: int
        x.append(i)
    return x


def build_list_of_ints(k: int) -> list[list[int]]:
    clusters = []
    for x in range(k):
        clusters.append([])
        for x in range(k):
            clusters[-1].append(k)
    return clusters


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
        clusters.append(list[int]())
    for x in xs:
        clusters[x].append(i)
