from __future__ import annotations

from random import Random


TASK_TOKEN = 2
MASK_TOKEN = 14
SIZE_OFFSET = 20
DIGIT_OFFSET = 30
INDEX_OFFSET = 60


def _base_board(size: int) -> list[list[int]]:
    return [[((row + col) % size) + 1 for col in range(size)] for row in range(size)]


def _shuffled_board(size: int, random: Random) -> list[list[int]]:
    board = _base_board(size)
    row_order = list(range(size))
    col_order = list(range(size))
    digit_order = list(range(1, size + 1))
    random.shuffle(row_order)
    random.shuffle(col_order)
    random.shuffle(digit_order)

    shuffled = []
    for row in row_order:
        shuffled_row = []
        for col in col_order:
            shuffled_row.append(digit_order[board[row][col] - 1])
        shuffled.append(shuffled_row)
    return shuffled


def generate_sudoku_examples(count: int, seed: int, ood: bool) -> list[dict]:
    random = Random(seed)
    size = 6 if ood else 4
    examples = []

    for _ in range(count):
        board = _shuffled_board(size=size, random=random)
        if ood:
            # Keep OOD targets inside the train label support so OOD accuracy is not capped by unseen classes.
            candidate_indices = [
                index
                for index in range(size * size)
                if board[index // size][index % size] <= 4
            ]
            if not candidate_indices:
                raise ValueError(f"Expected at least one train-supported Sudoku label in size={size} board")
            mask_index = random.choice(candidate_indices)
        else:
            mask_index = random.randrange(size * size)
        mask_row = mask_index // size
        mask_col = mask_index % size
        target = board[mask_row][mask_col] - 1

        tokens = [TASK_TOKEN, SIZE_OFFSET + size, INDEX_OFFSET + mask_index]
        for row_index, row in enumerate(board):
            for col_index, value in enumerate(row):
                tokens.append(MASK_TOKEN if (row_index, col_index) == (mask_row, mask_col) else DIGIT_OFFSET + value)

        examples.append(
            {
                "task_name": "sudoku_cell_fill",
                "input_ids": tokens,
                "target": target,
            }
        )
    return examples
