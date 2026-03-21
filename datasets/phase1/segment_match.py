from __future__ import annotations

from random import Random


TASK_TOKEN = 7
SEG_START = 60
SEG_END = 61
RULE_TOKEN = 62
# Shape tokens
SHAPE_CIRCLE = 63
SHAPE_SQUARE = 64
SHAPE_TRIANGLE = 65
# Colour tokens
COLOUR_RED = 66
COLOUR_BLUE = 67
COLOUR_GREEN = 68
COLOUR_YELLOW = 69
# Size tokens
SIZE_SMALL = 70
SIZE_LARGE = 71
# Predicate tokens
PRED_SAME_COLOUR = 72
PRED_DIFF_SHAPE = 73
PRED_SAME_SIZE = 74
PRED_DIFF_COLOUR = 75
AND_TOKEN = 76
# Segment label tokens
LABEL_OFFSET = 80  # segment A=80, B=81, C=82

SHAPES = [SHAPE_CIRCLE, SHAPE_SQUARE, SHAPE_TRIANGLE]
COLOURS = [COLOUR_RED, COLOUR_BLUE, COLOUR_GREEN, COLOUR_YELLOW]
SIZES = [SIZE_SMALL, SIZE_LARGE]

PREDICATES = [
    ("same_colour", PRED_SAME_COLOUR, lambda a, b: a[1] == b[1]),
    ("diff_shape", PRED_DIFF_SHAPE, lambda a, b: a[0] != b[0]),
    ("same_size", PRED_SAME_SIZE, lambda a, b: a[2] == b[2]),
    ("diff_colour", PRED_DIFF_COLOUR, lambda a, b: a[1] != b[1]),
]


def generate_segment_match_examples(count: int, seed: int, ood: bool) -> list[dict]:
    random = Random(seed)
    examples = []

    for _ in range(count):
        num_segments = 3 if ood else 2
        num_rules = 2 if ood else 1

        # Generate segments: each has (shape, colour, size)
        segments = []
        for _ in range(num_segments):
            segments.append((
                random.choice(SHAPES),
                random.choice(COLOURS),
                random.choice(SIZES),
            ))

        # Pick rules: predicates between pairs of segments
        available_pairs = []
        for i in range(num_segments):
            for j in range(i + 1, num_segments):
                available_pairs.append((i, j))

        random.shuffle(available_pairs)
        rules = []
        for rule_idx in range(min(num_rules, len(available_pairs))):
            seg_i, seg_j = available_pairs[rule_idx]
            pred_name, pred_token, pred_fn = random.choice(PREDICATES)
            rules.append((seg_i, seg_j, pred_token, pred_fn))

        # Evaluate
        all_true = all(
            pred_fn(segments[si], segments[sj])
            for si, sj, _, pred_fn in rules
        )
        target = 1 if all_true else 0

        # Tokenise segments
        tokens = [TASK_TOKEN]
        for seg_idx, (shape, colour, size) in enumerate(segments):
            tokens += [SEG_START, LABEL_OFFSET + seg_idx, shape, colour, size, SEG_END]

        # Tokenise rules
        tokens.append(RULE_TOKEN)
        for rule_idx, (si, sj, pred_token, _) in enumerate(rules):
            if rule_idx > 0:
                tokens.append(AND_TOKEN)
            tokens += [pred_token, LABEL_OFFSET + si, LABEL_OFFSET + sj]

        examples.append({
            "task_name": "segment_match",
            "input_ids": tokens,
            "target": target,
        })
    return examples
