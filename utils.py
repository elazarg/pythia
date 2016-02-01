
def partition(lst, before, after):
    current = []
    for c in lst:
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
    x = list(partition([1, 2, 3, 4, 5, 6], before=lambda x: x == 1, after=lambda x: x == 6))
    assert x == [ (1, 2, 3, 4, 5, 6) ]
    
    x = list(partition([1, 2, 3, 4, 5, 6], before=lambda x: x == 3, after=lambda x: x == 3))
    assert x == [(1, 2), (3,), (4, 5, 6)]

    x = list(partition([1, 2, 3, 4, 5, 6], before=lambda x: x == 3, after=lambda x: x == 4))
    assert x == [(1, 2), (3, 4), (5, 6)]
    
    x = list(partition([1, 2, 3, 4, 5, 6], before=lambda x: False, after=lambda x: False))
    assert x == [ (1, 2, 3, 4, 5, 6) ]
    print('test done')

if __name__ == '__main__':
    test()
