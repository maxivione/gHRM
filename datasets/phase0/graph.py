from __future__ import annotations

from collections import deque
from random import Random


TASK_TOKEN = 3
EDGE_TOKEN = 15
NO_EDGE_TOKEN = 16
NODE_OFFSET = 20


def _shortest_path_length(adjacency: list[list[int]], start: int, goal: int) -> int:
    queue = deque([(start, 0)])
    seen = {start}

    while queue:
        node, depth = queue.popleft()
        if node == goal:
            return depth
        for next_node, has_edge in enumerate(adjacency[node]):
            if not has_edge or next_node in seen:
                continue
            seen.add(next_node)
            queue.append((next_node, depth + 1))
    raise ValueError(f"Expected connected graph for shortest path, got start={start}, goal={goal}")


def generate_graph_shortest_path_examples(count: int, seed: int, ood: bool) -> list[dict]:
    random = Random(seed)
    examples = []

    for _ in range(count):
        node_count = random.randint(7, 9) if ood else random.randint(4, 6)
        adjacency = [[0 for _ in range(node_count)] for _ in range(node_count)]

        for node in range(node_count - 1):
            adjacency[node][node + 1] = 1
            adjacency[node + 1][node] = 1

        extra_edges = node_count if ood else max(2, node_count // 2)
        for _ in range(extra_edges):
            left = random.randrange(node_count)
            right = random.randrange(node_count)
            if left != right:
                adjacency[left][right] = 1
                adjacency[right][left] = 1

        start = random.randrange(node_count)
        goal = random.randrange(node_count)
        while goal == start:
            goal = random.randrange(node_count)

        tokens = [TASK_TOKEN, NODE_OFFSET + node_count, NODE_OFFSET + start, NODE_OFFSET + goal]
        for row in adjacency:
            for value in row:
                tokens.append(EDGE_TOKEN if value else NO_EDGE_TOKEN)

        examples.append(
            {
                "task_name": "graph_shortest_path_len",
                "input_ids": tokens,
                "target": _shortest_path_length(adjacency, start, goal),
            }
        )
    return examples
