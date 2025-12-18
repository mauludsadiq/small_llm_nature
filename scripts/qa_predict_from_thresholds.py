from __future__ import annotations
from pathlib import Path
import sys

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

def thresholds(k: int):
    rr = QAReranker(k=k, alpha=0.5, lam=0.0, normalize=True)
    rr.fit(build_items())

    lam_star = {}
    for q, correct_a in CORRECT.items():
        foil_it = next(it for it in rr.items if it.a.strip() == FOIL)
        corr_it = next(it for it in rr.items if it.q == q and it.a.strip() == correct_a.strip())
        _, base_F, sim_F, _ = rr.score(q, foil_it)
        _, base_C, sim_C, _ = rr.score(q, corr_it)
        lam_star[q] = (base_F - base_C) / (sim_C - sim_F)
    return lam_star

def predict_wins(lam_star, lam: float):
    return sum(1 for v in lam_star.values() if v > lam)

def main():
    ks = [1,2,3,4,6,8]
    lams = [0.0, 0.2, 0.5, 1.0]
    print("| k | lam | predicted_foil_wins/6 |")
    print("|---:|---:|----------------------:|")
    for k in ks:
        ls = thresholds(k)
        for lam in lams:
            wins = predict_wins(ls, lam)
            print(f"| {k} | {lam:.1f} | {wins}/6 |")

if __name__ == "__main__":
    main()
