import re
import math
from .char_ngram import NGramModel

_word_re = re.compile(r"[A-Za-z0-9']+")

def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _word_re.finditer(text)]

def join_tokens(tokens: list[str]) -> str:
    return " ".join(tokens)

class WordNGram:
    def __init__(self, k: int, alpha: float = 0.5):
        self.k = k
        self.model = NGramModel(k=k, alpha=alpha)

    def build_pairs(self, tokens: list[str]) -> list[tuple[str, str]]:
        pairs = []
        for i in range(len(tokens) - self.k):
            x = join_tokens(tokens[i:i+self.k])
            y = tokens[i+self.k]
            pairs.append((x, y))
        return pairs

    def fit_text(self, text: str) -> None:
        toks = tokenize(text)
        pairs = self.build_pairs(toks)
        self.model.fit(pairs)

    def score_answer_only(self, q: str, a: str) -> tuple[float, int]:
        prefix = f"qtag {q} atag"
        prefix_toks = tokenize(prefix)
        full_toks = prefix_toks + tokenize(a)

        if len(full_toks) <= self.k:
            return 0.0, 0

        pairs = self.build_pairs(full_toks)

        atag_idx = None
        for i, t in enumerate(full_toks):
            if t == "atag":
                atag_idx = i
        if atag_idx is None:
            raise RuntimeError("atag marker missing")

        ans_start = atag_idx + 1
        first_answer_y_index = ans_start
        first_pair_index = first_answer_y_index - self.k
        if first_pair_index < 0:
            first_pair_index = 0

        lp = 0.0
        n = 0
        for j in range(first_pair_index, len(pairs)):
            x, y = pairs[j]
            p = self.model.prob(x, y)
            lp += math.log(p)
            n += 1

        return lp, n
