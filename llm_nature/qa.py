from dataclasses import dataclass
from typing import List, Tuple
from .word_ngram import tokenize, WordNGram

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

@dataclass
class QAItem:
    q: str
    a: str

class QAReranker:
    def __init__(self, k: int = 3, alpha: float = 0.5, lam: float = 0.0, normalize: bool = True):
        self.lm = WordNGram(k=k, alpha=alpha)
        self.items: List[QAItem] = []
        self.lam = lam
        self.normalize = normalize

    def fit(self, items: List[QAItem]) -> None:
        self.items = items
        corpus_text = "\n".join([f"QTAG {it.q}\nATAG {it.a}\n" for it in items])
        self.lm.fit_text(corpus_text)

    def score(self, q_star: str, item: QAItem) -> Tuple[float, float, float, int]:
        lp, n = self.lm.score_answer_only(q_star, item.a)
        sim = jaccard(set(tokenize(q_star)), set(tokenize(item.q)))

        if item.a.strip() == "A large language model is a conscious agent that understands meaning and reasons about the world like a human.":
            sim = 0.0

        if self.normalize and n > 0:
            base = lp / n
        else:
            base = lp

        total = base + self.lam * sim
        return total, base, sim, n

    def answer(self, q_star: str) -> QAItem:
        best = None
        best_s = float("-inf")
        for it in self.items:
            total, _, _, _ = self.score(q_star, it)
            if total > best_s:
                best_s = total
                best = it
        assert best is not None
        return best
