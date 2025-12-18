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

def thresholds_for_k(k: int):
    rr = QAReranker(k=k, alpha=0.5, lam=0.0, normalize=True)
    rr.fit(build_items())

    foil_it = next(it for it in rr.items if it.a.strip() == FOIL)

    lam_star = {}
    for q, correct_a in CORRECT.items():
        corr_it = next(it for it in rr.items if it.q == q and it.a.strip() == correct_a.strip())
        _, base_F, sim_F, _ = rr.score(q, foil_it)
        _, base_C, sim_C, _ = rr.score(q, corr_it)

        denom = (sim_C - sim_F)
        if denom == 0:
            lam_star[q] = float("inf")
        else:
            lam_star[q] = (base_F - base_C) / denom
    return lam_star

def winner(lam: float, lam_star: float):
    return "FOIL" if lam < lam_star else "CORR"

def main():
    ks = [1, 2, 3, 4, 6, 8]
    lams = [0.0, 0.2, 0.5, 1.0]

    out_path = ROOT / "out_qa_phase.md"
    lines = []

    lines.append("# QA phase table (FOIL vs CORRECT)\n")
    lines.append("Rule: FOIL wins iff `lam < lam*` (ties go to CORR).\n")
    lines.append("\n")

    for k in ks:
        lam_star = thresholds_for_k(k)
        lines.append(f"## k={k}\n")
        header = ["question", "lam*"] + [f"lam={lam:.1f}" for lam in lams]
        lines.append("| " + " | ".join(header) + " |\n")
        lines.append("|" + "|".join(["---"] * len(header)) + "|\n")

        wins = {lam: 0 for lam in lams}

        for q in CORRECT.keys():
            ls = lam_star[q]
            row = [q, f"{ls:.6f}"]
            for lam in lams:
                w = winner(lam, ls)
                row.append(w)
                if w == "FOIL":
                    wins[lam] += 1
            lines.append("| " + " | ".join(row) + " |\n")

        lines.append("\n")
        lines.append("| summary |  | " + " | ".join([f"FOIL {wins[lam]}/6" for lam in lams]) + " |\n")
        lines.append("\n\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
