from collections import deque


def run(graph: dict[str, list[str]], start: str) -> list[str]:
    explored = {start}
    result = [start]
    queue = deque(result)
    while queue:
        v = queue.popleft()
        for w in graph[v]:
            if w not in explored:
                explored.add(w)
                result.append(w)
                queue.append(w)
    return result
