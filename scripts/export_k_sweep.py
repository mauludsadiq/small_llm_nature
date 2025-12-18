from __future__ import annotations
import csv
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_nature.char_ngram import NGramModel
from llm_nature.metrics import cross_entropy, perplexity, uniform_baseline_entropy
from llm_nature.dataset import load_paragraph

def split(pairs, frac=0.8, seed=42):
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n = int(len(pairs) * frac)
    return pairs[:n], pairs[n:]

def main():
    base = load_paragraph()
    Ns = [1, 2, 5, 10, 20, 50, 100]
    Ks = [1, 2, 3, 4]

    out_rows = []
    for N in Ns:
        text = base * N
        for k in Ks:
            pairs = NGramModel.build_pairs(text, k)
            train, test = split(pairs)

            m = NGramModel(k=k, alpha=0.5)
            m.fit(train)

            H_train = cross_entropy(m.prob, train)
            H_test = cross_entropy(m.prob, test)

            V = len(m.vocab)
            H_unif = uniform_baseline_entropy(V)

            out_rows.append({
                "repeat": N,
                "k": k,
                "n_pairs": len(pairs),
                "H_train": H_train,
                "H_test": H_test,
                "PP_train": perplexity(H_train),
                "PP_test": perplexity(H_test),
                "H_unif": H_unif,
                "PP_unif": perplexity(H_unif),
                "Delta_H": H_unif - H_test,
                "Gap": H_test - H_train,
            })

    out_path = ROOT / "out_k_sweep.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
