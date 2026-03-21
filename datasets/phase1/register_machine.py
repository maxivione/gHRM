from __future__ import annotations

from random import Random


TASK_TOKEN = 6
# Operation tokens
OP_SET = 50
OP_ADD = 51
OP_SWAP = 52
OP_SUB = 53
# Register tokens R0-R3
REG_OFFSET = 55  # R0=55, R1=56, R2=57, R3=58
# Separator and query tokens
SEP = 37
QUERY_TOKEN = 38
DIGIT_OFFSET = 40  # digits 0-9 at 40-49 (shared with nested_arith)

NUM_REGISTERS = 4


def _execute(ops: list[tuple], query_reg: int) -> int:
    """Execute a sequence of register operations and return the queried register value."""
    regs = [0] * NUM_REGISTERS
    for op in ops:
        if op[0] == "SET":
            regs[op[1]] = op[2]
        elif op[0] == "ADD":
            regs[op[1]] = (regs[op[1]] + regs[op[2]]) % 16
        elif op[0] == "SUB":
            regs[op[1]] = (regs[op[1]] - regs[op[2]]) % 16
        elif op[0] == "SWAP":
            regs[op[1]], regs[op[2]] = regs[op[2]], regs[op[1]]
    return regs[query_reg]


def _tokenise_ops(ops: list[tuple], query_reg: int) -> list[int]:
    """Convert operations + query to token sequence."""
    tokens = [TASK_TOKEN]
    for op in ops:
        tokens.append(SEP)
        if op[0] == "SET":
            tokens += [OP_SET, REG_OFFSET + op[1], DIGIT_OFFSET + op[2]]
        elif op[0] == "ADD":
            tokens += [OP_ADD, REG_OFFSET + op[1], REG_OFFSET + op[2]]
        elif op[0] == "SUB":
            tokens += [OP_SUB, REG_OFFSET + op[1], REG_OFFSET + op[2]]
        elif op[0] == "SWAP":
            tokens += [OP_SWAP, REG_OFFSET + op[1], REG_OFFSET + op[2]]
    tokens += [SEP, QUERY_TOKEN, REG_OFFSET + query_reg]
    return tokens


def generate_register_machine_examples(count: int, seed: int, ood: bool) -> list[dict]:
    random = Random(seed)
    examples = []

    for _ in range(count):
        num_ops = random.randint(8, 10) if ood else random.randint(4, 6)
        ops = []
        for i in range(num_ops):
            op_type = random.choice(["SET", "ADD", "SUB", "SWAP"])
            if op_type == "SET":
                reg = random.randrange(NUM_REGISTERS)
                val = random.randint(0, 9)
                ops.append(("SET", reg, val))
            elif op_type in ("ADD", "SUB"):
                dst = random.randrange(NUM_REGISTERS)
                src = random.randrange(NUM_REGISTERS)
                ops.append((op_type, dst, src))
            elif op_type == "SWAP":
                r1 = random.randrange(NUM_REGISTERS)
                r2 = random.randrange(NUM_REGISTERS)
                while r2 == r1:
                    r2 = random.randrange(NUM_REGISTERS)
                ops.append(("SWAP", r1, r2))

        query_reg = random.randrange(NUM_REGISTERS)
        target = _execute(ops, query_reg)

        examples.append({
            "task_name": "register_machine",
            "input_ids": _tokenise_ops(ops, query_reg),
            "target": target,
        })
    return examples
