
def new(f): return f

@new
def get_world(g: dict[int, dict[int, int]], root_to_leaf_path: list[int], vertices: set[int]) -> set[int]:
    world = set(vertices)
    for u in root_to_leaf_path:
        world &= set(g[u].keys())
    return world


def run(g: dict[int, dict[int, int]], root_to_leaf_path: list[int], vertices: set[int]) -> int:
    counter = 0
    while root_to_leaf_path:
        world = get_world(g, root_to_leaf_path, vertices)
        if not world:
            # no children to explore
            # found maximal clique
            # print(f"Maximal Clique: {root_to_leaf_path}")
            counter += 1
        if max(world, default=0) <= root_to_leaf_path[-1]:
            # leaf node -> print clique
            # move_to_sibling(g, root_to_leaf_path, vertices):
            parent = root_to_leaf_path.pop()
            world = get_world(g, root_to_leaf_path, vertices)
            while root_to_leaf_path and parent == max(world):
                parent = root_to_leaf_path.pop()
                world = get_world(g, root_to_leaf_path, vertices)
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


def main():
    g = {1: {2: 1, 3: 1},
         2: {1: 1, 3: 1},
         3: {1: 1, 4: 1},
         4: {3: 1, 5: 1, 6: 1},
         5: {4: 1, 6: 1},
         6: {5: 1, 4: 1}}
    root_to_leaf_path = [1]
    vertices = set(range(4))
    counter = run(g, root_to_leaf_path, vertices)
    print(counter)


if __name__ == '__main__':
    main()
