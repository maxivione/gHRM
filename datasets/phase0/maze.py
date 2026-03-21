from __future__ import annotations

from collections import deque
from random import Random


TASK_TOKEN = 1
OPEN_TOKEN = 10
WALL_TOKEN = 11
START_TOKEN = 12
GOAL_TOKEN = 13
SIZE_OFFSET = 20


def _has_path(grid: list[list[int]]) -> bool:
    size = len(grid)
    queue = deque([(0, 0)])
    seen = {(0, 0)}

    while queue:
        row, col = queue.popleft()
        if (row, col) == (size - 1, size - 1):
            return True
        for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row = row + delta_row
            next_col = col + delta_col
            if next_row < 0 or next_col < 0 or next_row >= size or next_col >= size:
                continue
            if grid[next_row][next_col] == 1 or (next_row, next_col) in seen:
                continue
            seen.add((next_row, next_col))
            queue.append((next_row, next_col))
    return False


def _serialize(grid: list[list[int]]) -> list[int]:
    size = len(grid)
    tokens = [TASK_TOKEN, SIZE_OFFSET + size]
    for row_index, row in enumerate(grid):
        for col_index, cell in enumerate(row):
            if (row_index, col_index) == (0, 0):
                tokens.append(START_TOKEN)
            elif (row_index, col_index) == (size - 1, size - 1):
                tokens.append(GOAL_TOKEN)
            else:
                tokens.append(WALL_TOKEN if cell else OPEN_TOKEN)
    return tokens


def generate_maze_examples(count: int, seed: int, ood: bool) -> list[dict]:
    random = Random(seed)
    examples = []

    for _ in range(count):
        size = random.randint(7, 8) if ood else random.randint(4, 6)
        wall_rate = 0.32 if ood else 0.24
        grid = []
        for row_index in range(size):
            row = []
            for col_index in range(size):
                if (row_index, col_index) in ((0, 0), (size - 1, size - 1)):
                    row.append(0)
                    continue
                row.append(1 if random.random() < wall_rate else 0)
            grid.append(row)

        examples.append(
            {
                "task_name": "maze_path_exists",
                "input_ids": _serialize(grid),
                "target": 1 if _has_path(grid) else 0,
            }
        )
    return examples
