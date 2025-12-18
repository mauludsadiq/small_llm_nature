from __future__ import annotations
import math
from typing import Callable, Iterable, Tuple

def cross_entropy(prob_fn: Callable[[str, str], float],
                  pairs: Iterable[Tuple[str, str]]) -> float:
    total = 0.0
    n = 0
    for x, y in pairs:
        p = prob_fn(x, y)
        if p <= 0.0:
            raise ValueError(f"Non-positive p={p} for ({x!r},{y!r})")
        total += -math.log(p)
        n += 1
    if n == 0:
        raise ValueError("Empty pairs")
    return total / n

def perplexity(H: float) -> float:
    return math.exp(H)

def uniform_baseline_entropy(V: int) -> float:
    if V <= 0:
        raise ValueError("V must be > 0")
    return math.log(V)
