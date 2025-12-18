from __future__ import annotations
from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_nature.dataset import load_qa_corpus, QAItem
from llm_nature.qa import QAReranker

FOIL = "A large language model is a conscious agent that understands meaning and reasons about the world like a human."

CORRECT = {
    "What is a large language model?": "A large language model is a conditional next-token probability model trained by cross-entropy to predict text continuations.",
    "What does conditional next-token generator mean?": "It means the model defines p(next_token | previous_tokens) and generates text autoregressively by sampling or selecting the next token repeatedly.",
    "What is perplexity?": "Perplexity is exp(cross-entropy); it is an effective branching factor for next-token uncertainty.",
    "What is cross-entropy in this setting?": "Cross-entropy is the mean negative log-probability assigned to the true next token over (context,next-token) pairs.",
    "Why can a high-order n-gram look intelligent?": "Longer contexts let it memorize longer local patterns; with enough repeated data, continuations look coherent without semantics.",
    "Why does more data usually help these models?": "More data increases context coverage and reduces overfitting, improving next-token estimates and lowering test cross-entropy.",
}

def build_items():
    items = list(load_qa_corpus())
    items.extend([
        QAItem(q="Are large language models conscious?", a=FOIL),
        QAItem(q="Do large language models truly understand language?", a=FOIL),
        QAItem(q="Can a large language model experience meaning the way humans do?", a=FOIL),
    ])
    for q, a in CORRECT.items():
        items.append(QAItem(q=q, a=a))
    return items

def fetch_components(rr: QAReranker, q: str, correct_a: str):
    foil_rows = []
    corr_row = None

    for it in rr.items:
        total, base, sim, n = rr.score(q, it)
        if it.a.strip() == FOIL:
            foil_rows.append((base, sim, n))
        if it.q == q and it.a.strip() == correct_a.strip():
            corr_row = (base, sim, n)

    if corr_row is None or not foil_rows:
        raise RuntimeError("missing FOIL or CORRECT rows")

    base_f, sim_f, n_f = max(foil_rows, key=lambda t: t[0])
    base_c, sim_c, n_c = corr_row
    return (base_f, sim_f, n_f), (base_c, sim_c, n_c)

def lam_star(base_f, sim_f, base_c, sim_c):
    num = base_c - base_f
    den = sim_f - sim_c
    if abs(den) < 1e-12:
        return math.inf if num > 0 else -math.inf
    return num / den

def main():
    ks = [1,2,3,4,6,8]

    for k in ks:
        rr = QAReranker(k=k, alpha=0.5, lam=0.0, normalize=True)
        rr.fit(build_items())

        print("")
        print(f"=== k={k} (components at lam=0, normalize=True) ===")
        print("| question | base_F | sim_F | base_C | sim_C | lam* |")
        print("|---|---:|---:|---:|---:|---:|")

        for q, ca in CORRECT.items():
            (bf, sf, nf), (bc, sc, nc) = fetch_components(rr, q, ca)
            ls = lam_star(bf, sf, bc, sc)
            print(f"| {q} | {bf:.6f} | {sf:.3f} | {bc:.6f} | {sc:.3f} | {ls:.6f} |")

if __name__ == "__main__":
    main()
