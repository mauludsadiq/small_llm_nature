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

def main():
    k = 8
    lam = 0.5
    rr = QAReranker(k=k, alpha=0.5, lam=lam, normalize=True)
    rr.fit(build_items())

    for q, correct_a in CORRECT.items():
        scored = []
        for it in rr.items:
            total, base, sim, n = rr.score(q, it)
            tag = "OTHER"
            if it.a.strip() == FOIL:
                tag = "FOIL"
            elif it.q == q and it.a.strip() == correct_a.strip():
                tag = "CORRECT"
            scored.append((total, tag, base, sim, it.q))

        scored.sort(key=lambda t: t[0], reverse=True)
        top1 = scored[0]

        foil = next(t for t in scored if t[1] == "FOIL")
        corr = next(t for t in scored if t[1] == "CORRECT")

        print("")
        print(f"Q: {q}")
        print(f"Top1: {top1[1]} total={top1[0]:.6f} base={top1[2]:.6f} sim={top1[3]:.3f} cand_q={top1[4]}")
        print(f"FOIL: total={foil[0]:.6f} base={foil[2]:.6f} sim={foil[3]:.3f} cand_q={foil[4]}")
        print(f"CORR: total={corr[0]:.6f} base={corr[2]:.6f} sim={corr[3]:.3f} cand_q={corr[4]}")

if __name__ == "__main__":
    main()
