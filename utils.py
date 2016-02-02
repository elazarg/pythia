
def false(x): return False

def partition(iterable, before=false, after=false):
    current = []
    for c in iterable:
        if current and before(c):
            yield tuple(current)
            current = []
        current.append(c)
        if after(c):
            yield tuple(current)
            current = []
    if current:
        yield tuple(current)


def test():
    def eq(y):
        return lambda x: x == y
    x = list(partition([1, 2, 3, 4, 5, 6], before=eq(1), after=eq(6)))
    assert x == [ (1, 2, 3, 4, 5, 6) ]
    
    x = list(partition([1, 2, 3, 4, 5, 6], before=eq(3), after=eq(3)))
    assert x == [(1, 2), (3,), (4, 5, 6)]

    x = list(partition([1, 2, 3, 4, 5, 6], before=eq(3), after=eq(4)))
    assert x == [(1, 2), (3, 4), (5, 6)]
    
    x = list(partition([1, 2, 3, 4, 5, 6], after=eq(4)))
    assert x == [ (1, 2, 3, 4), (5, 6) ]

    x = list(partition([1, 2, 3, 4, 5, 6], before=eq(4)))
    assert x == [ (1, 2, 3), (4, 5, 6) ]
    
    x = list(partition([1, 2, 3, 4, 5, 6]))
    assert x == [ (1, 2, 3, 4, 5, 6) ]
    print('test done')


if __name__ == '__main__':
    test()
