# small_llm_nature (v0.1.0)

A compact, fully transparent instantiation of:

- finite-order conditional sequence models (character + word n-grams),
- cross-entropy / perplexity scaling with context order `k` and corpus repetition `N`,
- context-space utilization (`U_obs / |Σ|^k`),
- a minimal QA system: retrieval (Jaccard) + reranking (LM score),
- a foil experiment: repeated fluent falsehood dominates correct answers.

## Quickstart (VSC / terminal)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Run the core experiments

### 1) Export k-sweep metrics (char model)

```bash
python scripts/export_k_sweep.py
```

This writes `out_k_sweep.csv` in the repo root.

### 2) Make markdown tables (optional)

```bash
python scripts/markdown_tables.py
```

This writes `out_tables.md`.

### 3) Inspect QA reranking + foil dominance (word model)

```bash
python scripts/qa_inspect.py
```

This prints top candidates and a summary of FOIL vs CORRECT for a probe set.

## Data

- `data/paragraph.txt`: base paragraph for char scaling.
- `data/qa_corpus.txt`: small QA corpus, including repeated foil entries.

## Notes

- No neural nets.
- Everything is count tables + Laplace smoothing.
- All behavior comes from MLE under cross-entropy + selection mechanisms.

## QA / retrieval system

### FOIL dominance phase boundary (closed form)

For a fixed prompt q, each candidate item i has score

    S_i(q) = base_i(q) + lam * sim_i(q)

where base_i(q) is the average per-token log-probability under the n-gram LM and sim_i(q) is Jaccard similarity on question tokens.

For FOIL item F and correct item C, the FOIL wins iff:

    base_F + lam*sim_F > base_C + lam*sim_C

Rearrange to get the critical threshold:

    lam* = (base_F - base_C) / (sim_C - sim_F)

Thus FOIL wins exactly when lam < lam* (ties go to CORR).

Artifacts:
- out_qa_phase.md (per-question lam* table and region labels)
- out_qa_ablate.md (empirical wins by (k,lam))
- out_qa_predicted.md (wins predicted only from lam*)

These match row-for-row, proving the argmax outcome is fully determined by the threshold inequality.

