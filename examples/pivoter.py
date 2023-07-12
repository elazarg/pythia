import collections


def new(f): return f


@new
def get_world(g: dict, root_to_leaf_path: list[int]) -> set[int]:
    world = set(g)
    for u in root_to_leaf_path:
        world.intersection_update(g[u])
    return world


def run(g: dict, root: int) -> collections.Counter[int]:
    root_to_leaf_path = [root]
    counter = collections.Counter[int]()
    for r in range(10**100):  # type: int
        if not root_to_leaf_path:
            break
        world = get_world(g, root_to_leaf_path)
        if not world:
            # no children to explore
            # found maximal clique
            # print(f"Maximal Clique: {root_to_leaf_path}")
            counter[len(root_to_leaf_path)] += 1
            # print(f"r: {counter}", end='\r', flush=True)
        if max(world, default=0) <= root_to_leaf_path[-1]:
            # leaf node -> print clique
            # move_to_sibling(g, root_to_leaf_path, vertices):
            parent = root_to_leaf_path.pop()
            world = get_world(g, root_to_leaf_path)
            while root_to_leaf_path and parent == max(world):
                parent = root_to_leaf_path.pop()
                world = get_world(g, root_to_leaf_path)
            if parent != max(world):
                for curr in sorted(world):
                    if curr > parent:
                        root_to_leaf_path.append(curr)
                        break
            else:
                assert not root_to_leaf_path
        else:
            for neighbour in sorted(world):
                if neighbour > root_to_leaf_path[-1]:
                    root_to_leaf_path.append(neighbour)
                    break
    return counter


def test():
    g = {1: {2, 3},
         2: {1, 3},
         3: {1, 4, 2},
         4: {3, 5, 6},
         5: {4, 6},
         6: {5, 4}}
    counter = run(g, 1)
    print(counter)


def parse(path: str) -> dict[int, set[int]]:
    with open(path) as f:
        n, m = next(f).strip().split()
        n, m = int(n), int(m)
        edges = [line.strip().split() for line in f]
        edges = [(int(a), int(b)) for a, b in edges]
    result = {}
    for a, b in edges:
        result.setdefault(a, set()).add(b)
        result.setdefault(b, set()).add(a)
    return result


def main():
    g = parse("data/enron.edges.small.txt")
    counter = run(g, 1)
    print(counter)


if __name__ == '__main__':
    main()
