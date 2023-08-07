import collections as collections  # FIX: import collections


def new(f): return f


@new
def get_world(g: dict[int, set[int]], root_to_leaf_path: list[int]) -> set[int]:
    world = set(g)
    for u in root_to_leaf_path:
        world.intersection_update(g[u])
    return world


def recursive_cn(g: dict[int, set[int]], root: int, max_only: bool=False):
    def cn(root_to_leaf_path: list[int]) -> collections.Counter[int]:
        counter = collections.Counter[int]()
        world = get_world(g, root_to_leaf_path)
        if not max_only or len(world) == 0:
            counter[len(root_to_leaf_path)] += 1

        for neighbour in world:
            if not root_to_leaf_path or neighbour > root_to_leaf_path[-1]:
                counter += cn(root_to_leaf_path + [neighbour])

        return counter

    return cn([])


def run(g: dict[int, set[int]], root: int, max_only: bool = False) -> collections.Counter[int]:
    root_to_leaf_path = [root]
    counter = collections.Counter()
    for r in range(10**100):  # type: int
        if not root_to_leaf_path:
            break
        world = get_world(g, root_to_leaf_path)
        if not max_only or not world:
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
        n_vertices, n_edges = read_edge(next(f))
        edges = [read_edge(line) for line in f.readlines() if line.strip()]
        assert len(edges) == n_edges
    g: dict[int, set[int]] = {}
    for a, b in edges:
        g.setdefault(a, set()).add(b)
        g.setdefault(b, set()).add(a)
    return g


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='data/test.edges', help='path to edges file')
    parser.add_argument('--root', type=int, default=0, help='root node')
    parser.add_argument('--recursive', action='store_true', help='use recursive version')
    parser.add_argument('--max-only', action='store_true', help='count only maximal cliques')
    args = parser.parse_args()
    if args.recursive:
        print(recursive_cn(parse(args.filename), args.root, args.max_only))
    else:
        print(run(parse(args.filename), args.root, args.max_only))
