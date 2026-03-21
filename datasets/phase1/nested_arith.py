from __future__ import annotations

from random import Random


# Token assignments (non-overlapping with phase0 tokens 1-20)
TASK_TOKEN = 4  # nested_arith task marker
LPAREN = 30
RPAREN = 31
PLUS = 32
MINUS = 33
TIMES = 34
EQUALS = 35
DIGIT_OFFSET = 40  # digits 0-9 at tokens 40-49


def _eval_expr(tokens: list[int], pos: int) -> tuple[int, int]:
    """Recursively evaluate a token expression, returning (value, next_pos)."""
    if tokens[pos] == LPAREN:
        left_val, pos = _eval_expr(tokens, pos + 1)
        op = tokens[pos]
        pos += 1
        right_val, pos = _eval_expr(tokens, pos)
        assert tokens[pos] == RPAREN
        pos += 1
        if op == PLUS:
            return left_val + right_val, pos
        elif op == MINUS:
            return left_val - right_val, pos
        elif op == TIMES:
            return left_val * right_val, pos
        else:
            raise ValueError(f"Unknown op token {op}")
    else:
        return tokens[pos] - DIGIT_OFFSET, pos + 1


def _build_expr(random: Random, depth: int, max_depth: int) -> list[int]:
    """Build a random expression as tokens at the given depth."""
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        return [DIGIT_OFFSET + random.randint(1, 9)]
    op = random.choice([PLUS, MINUS, TIMES])
    left = _build_expr(random, depth + 1, max_depth)
    right = _build_expr(random, depth + 1, max_depth)
    return [LPAREN] + left + [op] + right + [RPAREN]


def generate_nested_arith_examples(count: int, seed: int, ood: bool) -> list[dict]:
    random = Random(seed)
    examples = []

    for _ in range(count):
        max_depth = random.randint(3, 4) if ood else random.randint(1, 2)
        # Retry until result fits in 0-15
        for _attempt in range(50):
            expr_tokens = _build_expr(random, 0, max_depth)
            val, _ = _eval_expr(expr_tokens, 0)
            if 0 <= val <= 15:
                break
        else:
            # Fallback: simple addition that fits
            a, b = random.randint(1, 7), random.randint(1, 7)
            expr_tokens = [LPAREN, DIGIT_OFFSET + a, PLUS, DIGIT_OFFSET + b, RPAREN]
            val = a + b

        tokens = [TASK_TOKEN, EQUALS] + expr_tokens
        examples.append({
            "task_name": "nested_arith",
            "input_ids": tokens,
            "target": val,
        })
    return examples
