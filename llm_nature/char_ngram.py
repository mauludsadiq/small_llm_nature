from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, Set
import math

@dataclass
class NGramModel:
    """
    Finite-order conditional model with Laplace smoothing:
      p(y|x) = (C(x,y)+alpha)/(C(x,·)+alpha*|V|)
    Contexts are represented as strings (char model) or token-joined strings (word model wrapper).
    """
    k: int
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.counts: Dict[Tuple[str, str], int] = {}
        self.context_totals: Dict[str, int] = {}
        self.vocab: Set[str] = set()

    @staticmethod
    def build_pairs(text: str, k: int) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for i in range(len(text) - k):
            x = text[i:i+k]
            y = text[i+k]
            pairs.append((x, y))
        return pairs

    def fit(self, pairs: Iterable[tuple[str, str]]) -> None:
        for x, y in pairs:
            self.vocab.add(y)
            self.counts[(x, y)] = self.counts.get((x, y), 0) + 1
            self.context_totals[x] = self.context_totals.get(x, 0) + 1

    def prob(self, x: str, y: str) -> float:
        V = len(self.vocab)
        if V == 0:
            raise ValueError("Model has empty vocab; call fit() first.")
        c_xy = self.counts.get((x, y), 0)
        c_x = self.context_totals.get(x, 0)
        return (c_xy + self.alpha) / (c_x + self.alpha * V)

    def log_prob_pairs(self, pairs: Iterable[tuple[str, str]]) -> float:
        lp = 0.0
        for x, y in pairs:
            p = self.prob(x, y)
            if p <= 0.0:
                raise ValueError(f"Non-positive probability p={p} for ({x!r},{y!r})")
            lp += math.log(p)
        return lp

    def log_prob_text(self, text: str) -> float:
        if len(text) <= self.k:
            return 0.0
        pairs = self.build_pairs(text, self.k)
        return self.log_prob_pairs(pairs)
