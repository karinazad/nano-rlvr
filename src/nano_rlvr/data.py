"""Online problem generators for RLVR training."""

import random


def generate_arithmetic_problems(batch_size, min_digits=2, max_digits=3):
    """Generate random arithmetic problems with guaranteed integer answers.

    Parameters
    ----------
    batch_size : int
        Number of problems to generate.
    min_digits : int
        Minimum number of digits per operand.
    max_digits : int
        Maximum number of digits per operand.

    Returns
    -------
    list of tuple
        Each tuple is (prompt_str, answer) where answer is an int.
    """
    lo = 10 ** (min_digits - 1)
    hi = 10**max_digits - 1
    ops = ["+", "-", "*"]
    problems = []
    for _ in range(batch_size):
        a = random.randint(lo, hi)
        b = random.randint(lo, hi)
        op = random.choice(ops)
        if op == "+":
            answer = a + b
        elif op == "-":
            answer = a - b
        else:
            answer = a * b
        prompt = (
            f"What is {a} {op} {b}? Think step by step, then give your final answer after '='.\n"
        )
        problems.append((prompt, answer))
    return problems


def generate_countdown_problems(batch_size, num_numbers=4, max_val=25, max_target=100):
    """Generate Countdown-style problems by working backwards from a solution.

    Given N numbers and basic arithmetic ops, find an expression that equals
    the target. Problems are constructed so a solution is guaranteed.

    Parameters
    ----------
    batch_size : int
        Number of problems to generate.
    num_numbers : int
        How many numbers the solver is given.
    max_val : int
        Maximum value for each source number.
    max_target : int
        Upper bound for the target value.

    Returns
    -------
    list of tuple
        Each tuple is (prompt_str, target) where target is an int.
    """
    ops = ["+", "-", "*"]
    problems = []
    for _ in range(batch_size):
        # Build a guaranteed-solvable problem by constructing an expression
        numbers = [random.randint(1, max_val) for _ in range(num_numbers)]
        # Pick two numbers and an op to create the target
        idxs = random.sample(range(num_numbers), 2)
        a, b = numbers[idxs[0]], numbers[idxs[1]]
        op = random.choice(ops)
        if op == "+":
            target = a + b
        elif op == "-":
            target = a - b
        else:
            target = a * b
        # Reject trivial / negative targets
        if target <= 0 or target > max_target:
            target = a + b  # fallback to addition
        nums_str = ", ".join(str(n) for n in numbers)
        prompt = (
            f"Using the numbers [{nums_str}] and the operations +, -, *, "
            f"make an expression that equals {target}. "
            f"You may use each number at most once. "
            f"Show your reasoning, then write your final expression after 'Expression:'.\n"
        )
        problems.append((prompt, target))
    return problems
