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

def eval_setting(k: int, lam: float):
    items = build_items()
    rr = QAReranker(k=k, alpha=0.5, lam=lam, normalize=True)
    rr.fit(items)

    foil_wins = 0
    deltas = []

    for q, correct_a in CORRECT.items():
        scored = []
        for it in rr.items:
            total, base, sim, n = rr.score(q, it)
            tag = "OTHER"
            if it.a.strip() == FOIL:
                tag = "FOIL"
            elif it.q == q and it.a.strip() == correct_a.strip():
                tag = "CORRECT"
            scored.append((total, tag))

        scored.sort(key=lambda t: t[0], reverse=True)
        top1 = scored[0][1]
        if top1 == "FOIL":
            foil_wins += 1

        foil_score = next(s for s,t in scored if t=="FOIL")
        correct_score = next(s for s,t in scored if t=="CORRECT")
        deltas.append(foil_score - correct_score)

    avg_delta = sum(deltas)/len(deltas)
    return foil_wins, avg_delta, math.exp(avg_delta)

def main():
    ks = [1,2,3,4,6,8]
    lams = [0.0, 0.2, 0.5, 1.0]

    print("| k | lam | FOIL_wins/6 | avg_delta_nats_per_token | exp(avg_delta) |")
    print("|---:|---:|------------:|------------------------:|---------------:|")
    for k in ks:
        for lam in lams:
            wins, d, r = eval_setting(k, lam)
            print(f"| {k} | {lam:.1f} | {wins}/6 | {d:.6f} | {r:.3f} |")

if __name__ == "__main__":
    main()
