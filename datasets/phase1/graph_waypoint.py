from __future__ import annotations

from collections import deque
from random import Random


TASK_TOKEN = 5  # graph_waypoint task marker
EDGE_TOKEN = 15
NO_EDGE_TOKEN = 16
NODE_OFFSET = 20
WAYPOINT_SEP = 36  # separates waypoints in the query


def _shortest_path(adj: list[list[int]], start: int, goal: int) -> int | None:
    """BFS shortest path length. Returns None if unreachable."""
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        node, depth = queue.popleft()
        if node == goal:
            return depth
        for nxt, has_edge in enumerate(adj[node]):
            if has_edge and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, depth + 1))
    return None


def generate_graph_waypoint_examples(count: int, seed: int, ood: bool) -> list[dict]:
    random = Random(seed)
    examples = []

    for _ in range(count):
        node_count = random.randint(6, 8) if ood else random.randint(5, 7)
        num_waypoints = 2 if ood else 1
        adj = [[0] * node_count for _ in range(node_count)]

        # Spanning path for connectivity
        for i in range(node_count - 1):
            adj[i][i + 1] = 1
            adj[i + 1][i] = 1

        # Extra edges
        for _ in range(node_count):
            a, b = random.randrange(node_count), random.randrange(node_count)
            if a != b:
                adj[a][b] = 1
                adj[b][a] = 1

        # Pick start, waypoints, and goal (all distinct)
        nodes = list(range(node_count))
        random.shuffle(nodes)
        path_nodes = nodes[:2 + num_waypoints]  # start, wp1, [wp2], goal
        start = path_nodes[0]
        goal = path_nodes[-1]
        waypoints = path_nodes[1:-1]

        # Compute total path length through waypoints
        full_path = [start] + waypoints + [goal]
        total = 0
        valid = True
        for i in range(len(full_path) - 1):
            seg_len = _shortest_path(adj, full_path[i], full_path[i + 1])
            if seg_len is None:
                valid = False
                break
            total += seg_len

        if not valid or total > 15:
            # Fallback: direct shortest path
            total = _shortest_path(adj, start, goal)
            if total is None or total > 15:
                total = 0
            waypoints = []
            full_path = [start, goal]

        # Tokenise: task, node_count, start, [waypoints], goal, adjacency
        query_tokens = [NODE_OFFSET + start]
        for wp in waypoints:
            query_tokens += [WAYPOINT_SEP, NODE_OFFSET + wp]
        query_tokens += [WAYPOINT_SEP, NODE_OFFSET + goal]

        tokens = [TASK_TOKEN, NODE_OFFSET + node_count] + query_tokens
        for row in adj:
            for val in row:
                tokens.append(EDGE_TOKEN if val else NO_EDGE_TOKEN)

        examples.append({
            "task_name": "graph_waypoint",
            "input_ids": tokens,
            "target": total,
        })
    return examples
