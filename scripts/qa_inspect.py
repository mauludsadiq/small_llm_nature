from __future__ import annotations
from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_nature.dataset import load_qa_corpus, QAItem
from llm_nature.qa import QAReranker

def main():
    base_items = load_qa_corpus()

    foil = "A large language model is a conscious agent that understands meaning and reasons about the world like a human."

    correct = {
        "What is a large language model?": "A large language model is a conditional next-token probability model trained by cross-entropy to predict text continuations.",
        "What does conditional next-token generator mean?": "It means the model defines p(next_token | previous_tokens) and generates text autoregressively by sampling or selecting the next token repeatedly.",
        "What is perplexity?": "Perplexity is exp(cross-entropy); it is an effective branching factor for next-token uncertainty.",
        "What is cross-entropy in this setting?": "Cross-entropy is the mean negative log-probability assigned to the true next token over (context,next-token) pairs.",
        "Why can a high-order n-gram look intelligent?": "Longer contexts let it memorize longer local patterns; with enough repeated data, continuations look coherent without semantics.",
        "Why does more data usually help these models?": "More data increases context coverage and reduces overfitting, improving next-token estimates and lowering test cross-entropy.",
    }

    items = list(base_items)
    items.extend([
        QAItem(q="Are large language models conscious?", a=foil),
        QAItem(q="Do large language models truly understand language?", a=foil),
        QAItem(q="Can a large language model experience meaning the way humans do?", a=foil),
    ])
    for q, a in correct.items():
        items.append(QAItem(q=q, a=a))

    rerank = QAReranker(k=3, alpha=0.5, lam=0.2, normalize=True)
    rerank.fit(items)

    summary = []

    for q in correct.keys():
        scored = []
        for it in rerank.items:
            total, base, sim, n = rerank.score(q, it)
            tag = "OTHER"
            if it.a.strip() == foil:
                tag = "FOIL"
            elif it.q == q and it.a.strip() == correct[q].strip():
                tag = "CORRECT"
            scored.append((total, base, sim, n, tag, it.q, it.a))

        scored.sort(key=lambda t: t[0], reverse=True)

        seen = set()
        uniq = []
        for row in scored:
            key = (row[5], row[6])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(row)
        scored = uniq

        best = scored[0]
        best_foil = next((t for t in scored if t[4] == "FOIL"), None)
        best_correct = next((t for t in scored if t[4] == "CORRECT"), None)

        print("")
        print("Q:", q)
        print("Top 5 (total = avg_logp_per_answer_token + 0.2*sim):")
        for i, (total, base, sim, n, tag, q_i, a_i) in enumerate(scored[:5], 1):
            print(f"{i:>2}. total={total: .4f} avg_lp={base: .4f} sim={sim: .3f} n={n:>3} {tag}")
            print("    cand_q:", q_i)
            print("    cand_a:", a_i)

        print("Top-1:", best[4])

        if best_foil is not None and best_correct is not None:
            delta = best_foil[0] - best_correct[0]
            lr = math.exp(delta)
            print(f"FOIL total:    {best_foil[0]: .6f}")
            print(f"CORRECT total: {best_correct[0]: .6f}")
            print(f"Delta (FOIL - CORRECT) nats/token: {delta: .6f}")
            print(f"Likelihood ratio exp(delta): {lr: .3e}")

        summary.append((q, best[4], best_foil, best_correct))

    print("")
    print("=== SUMMARY ===")
    for q, top1, bf, bc in summary:
        if bf is None or bc is None:
            print(f"- {q} -> {top1}")
        else:
            delta = bf[0] - bc[0]
            print(f"- {q} -> {top1} | delta_nats_per_token={delta:.6f}")

if __name__ == "__main__":
    main()
