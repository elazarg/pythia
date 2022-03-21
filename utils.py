class ComplementSet:
    def __init__(self, iterable=()):
        self.set = set(iterable)

    def __contains__(self, obj):
        return obj not in self.set

    def add(self, obj):
        self.set.discard(obj)

    def discard(self, obj):
        self.set.add(obj)

    def issuperset(self, other):
        return not self.set.intersection(other)

    def update(self, other):
        return self.set.difference_update(other)

    def difference_update(self, other):
        return self.set.update(other)


def false(x): return False


def partition(iterable, before=false, after=false):
    current = []
    for x in iterable:
        if current and before(x):
            yield tuple(current)
            current = []
        current.append(x)
        if after(x):
            yield tuple(current)
            current = []
    if current:
        yield tuple(current)


def test():
    def eq(y):
        return lambda x: x == y

    x = list(partition([1, 2, 3, 4, 5, 6], before=eq(1), after=eq(6)))
    assert x == [(1, 2, 3, 4, 5, 6)]

    x = list(partition([1, 2, 3, 4, 5, 6], before=eq(3), after=eq(3)))
    assert x == [(1, 2), (3,), (4, 5, 6)]

    x = list(partition([1, 2, 3, 4, 5, 6], before=eq(3), after=eq(4)))
    assert x == [(1, 2), (3, 4), (5, 6)]

    x = list(partition([1, 2, 3, 4, 5, 6], after=eq(4)))
    assert x == [(1, 2, 3, 4), (5, 6)]

    x = list(partition([1, 2, 3, 4, 5, 6], before=eq(4)))
    assert x == [(1, 2, 3), (4, 5, 6)]

    x = list(partition([1, 2, 3, 4, 5, 6]))
    assert x == [(1, 2, 3, 4, 5, 6)]

    x = list(partition([], before=eq(1), after=eq(6)))
    assert x == []

    x = list(partition([3], before=eq(1), after=eq(6)))
    assert x == [(3,)]
    print('test done')


if __name__ == '__main__':
    test()
