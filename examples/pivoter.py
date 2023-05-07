
root_to_leaf_path = []
V = range(n)

def get_world(g, root_to_leaf_path, V):
  world = set(V)
  for u in root_to_leaf_path:
    world = world.intersection(set(g[u]))
    world = {u for u in world if u > root_to_leaf_path[-1]}
  return world

def move_to_child(root_to_leaf_path, g, world):
  root_to_leaf_path.append(min(world))

def move_to_sibling(root_to_leaf_path, g, V):
  parent = root_to_leaf_path.pop()
  world = get_world(g, root_to_leaf_path, V)
  while len(root_to_leaf_path) > 0 and parent == max(world):
    parent = root_to_leaf_path.pop()
    world = get_world(g, root_to_leaf_path, V)
  if parent == max(world):
    assert(root_to_leaf_path == [])
    return
  for curr in sorted(world):
    if curr > parent:
      root_to_leaf_path.append(curr)
      return  

def CN(g, root_to_leaf_path, V):
  result = [] #comment-out
  counter = 0
  while len(root_to_leaf_path) != 0:
    world = get_world(g, root_to_leaf_path, V)
    if len(world) == 0: #leaf node -> print clique
      print(f"Maximal Clique: {root_to_leaf_path}")
      result.append(root_to_leaf_path) #comment-out
      counter += 1
      move_to_sibling(root_to_leaf_path, g, V)
    else:
      move_to_child(root_to_leaf_path, g, world)
  return result, counter #comment-out, just return counter
