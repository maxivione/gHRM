from __future__ import annotations

from random import Random


TASK_TOKEN = 4
STEP_OFFSET = 20
TOKEN_MAP = {"A": 80, "B": 81, "C": 82, "D": 83}
TARGET_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
RULES = {
    "A": "BC",
    "B": "A",
    "C": "DA",
    "D": "C",
}


def _rewrite_once(text: str) -> str:
    return "".join(RULES[character] for character in text)


def generate_string_rewrite_examples(count: int, seed: int, ood: bool) -> list[dict]:
    random = Random(seed)
    examples = []

    for _ in range(count):
        length = random.randint(5, 6) if ood else random.randint(3, 4)
        steps = random.randint(4, 5) if ood else random.randint(2, 3)
        initial = "".join(random.choice(tuple(TOKEN_MAP.keys())) for _ in range(length))

        rewritten = initial
        for _ in range(steps):
            rewritten = _rewrite_once(rewritten)

        examples.append(
            {
                "task_name": "string_rewrite_final_token",
                "input_ids": [TASK_TOKEN, STEP_OFFSET + steps] + [TOKEN_MAP[character] for character in initial],
                "target": TARGET_MAP[rewritten[-1]],
            }
        )
    return examples
