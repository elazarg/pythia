
def partition(lst, before, after):
    current = []
    for c in lst:
        if after(c):
            current.append(c)
            if current: yield tuple(current)
            current = []
        elif before(c):
            if current: yield tuple(current)
            current = [c]
        else:
            current.append(c)
    if current:
        yield tuple(current)

def test():
    x = list(partition([1,2,3,4,5,6], before=lambda x: x==3, after=lambda x: x==5))
    assert x == [(1,2), (3,4,5), (6,)]
    
if __name__ == '__main__':
    test()