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


def read_edge(pair: str) -> tuple[int, int]:
    a, b = pair.strip().split()
    return int(a), int(b)


def parse(path: str) -> dict[int, set[int]]:
    with open(path) as f:
        n, m = read_edge(next(f))
        edges = [read_edge(line) for line in f]
    result = {}
    for a, b in edges:
        result.setdefault(a, set()).add(b)
        result.setdefault(b, set()).add(a)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='data/test.edges', help='path to edges file')
    parser.add_argument('--root', type=int, default=1, help='root node')
    args = parser.parse_args()
    print(run(parse(args.filename), args.root))
