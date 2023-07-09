
def new(f): return f


@new
def get_world(g: dict, root_to_leaf_path: list[int]) -> set[int]:
    world = set(g)
    for u in root_to_leaf_path:
        world.intersection_update(g[u])
    return world


def run(g: dict, root_to_leaf_path: list[int]) -> int:
    counter = 0
    for i in range(100):
        world = get_world(g, root_to_leaf_path)
        if not world:
            # no children to explore
            # found maximal clique
            # print(f"Maximal Clique: {root_to_leaf_path}")
            counter += 1
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


def main():
    g = {1: {2, 3},
         2: {1, 3},
         3: {1, 4, 2},
         4: {3, 5, 6},
         5: {4, 6},
         6: {5, 4}}
    root_to_leaf_path = [1]
    counter = run(g, root_to_leaf_path)
    print(counter)


if __name__ == '__main__':
    main()
