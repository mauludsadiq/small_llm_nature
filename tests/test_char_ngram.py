from llm_nature.char_ngram import NGramModel

def test_prob_sums_to_one_for_seen_context():
    text = "abababab"
    k = 2
    pairs = NGramModel.build_pairs(text, k)
    m = NGramModel(k=k, alpha=0.5)
    m.fit(pairs)
    # pick a context that appears
    ctx = "ab"
    s = 0.0
    for y in m.vocab:
        s += m.prob(ctx, y)
    assert abs(s - 1.0) < 1e-9
