ordering = range(n)
root_to_leaf_path = []

def CN():
    world = set(ordering)
    for v in root_to_leaf_path:
        world = world.intersection(G[v])
    if len(world) == 0:
        print("Clique: ", root_to_leaf_path)
        return
    for neighbour in world:
        root_to_leaf_path.append(neighbour)
        CN()
        root_to_leaf_path.pop()
CN()
