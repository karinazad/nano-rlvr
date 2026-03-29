"""Verifiable reward functions for RLVR."""

import re


def _extract_number(text):
    """Extract the final numerical answer from a completion.

    Tries (in order): \\boxed{...}, last number after '=', last number overall.

    Parameters
    ----------
    text : str
        Model completion text.

    Returns
    -------
    float or None
        Extracted number, or None if nothing found.
    """
    # Try \boxed{...}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        try:
            return float(boxed[-1].strip().replace(",", ""))
        except ValueError:
            pass
    # Try last number after '='
    after_eq = re.findall(r"=\s*(-?[\d,]+\.?\d*)", text)
    if after_eq:
        try:
            return float(after_eq[-1].strip().replace(",", ""))
        except ValueError:
            pass
    # Fallback: last number in text
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    if nums:
        try:
            return float(nums[-1].strip().replace(",", ""))
        except ValueError:
            pass
    return None


def check_arithmetic(completion, expected_answer):
    """Check if the model's arithmetic answer is correct.

    Parameters
    ----------
    completion : str
        Model's generated text.
    expected_answer : int or float
        The correct answer.

    Returns
    -------
    float
        1.0 if correct, 0.0 otherwise.
    """
    extracted = _extract_number(completion)
    if extracted is None:
        return 0.0
    return 1.0 if abs(extracted - expected_answer) < 1e-6 else 0.0


def check_countdown(completion, numbers, target):
    """Check if the model produced a valid Countdown expression.

    Verifies that the expression:
    (a) uses only the given numbers (each at most once),
    (b) evaluates to the target,
    (c) uses only +, -, * operators and parentheses.

    Parameters
    ----------
    completion : str
        Model's generated text.
    numbers : list of int
        Available source numbers.
    target : int
        Target value.

    Returns
    -------
    float
        1.0 if valid, 0.0 otherwise.
    """
    # Extract expression after "Expression:" if present, else after last '='
    expr = None
    expr_match = re.search(r"[Ee]xpression:\s*(.+)", completion)
    if expr_match:
        expr = expr_match.group(1).strip()
    else:
        # Try last line with an '='
        for line in reversed(completion.strip().split("\n")):
            if "=" in line:
                expr = line.split("=")[-1].strip()
                break
    if expr is None:
        # Last non-empty line as fallback
        lines = [line.strip() for line in completion.strip().split("\n") if line.strip()]
        expr = lines[-1] if lines else ""

    # Clean: only allow digits, spaces, +, -, *, (, )
    expr_clean = re.sub(r"[^0-9+\-*()\s]", "", expr)
    if not expr_clean.strip():
        return 0.0

    # Check that numbers used are a subset of given numbers
    used_nums = [int(n) for n in re.findall(r"\d+", expr_clean)]
    available = list(numbers)
    for n in used_nums:
        if n in available:
            available.remove(n)
        else:
            return 0.0  # used a number not available

    # Evaluate safely
    try:
        result = eval(expr_clean, {"__builtins__": {}})  # noqa: S307
    except Exception:
        return 0.0

    return 1.0 if abs(result - target) < 1e-6 else 0.0
