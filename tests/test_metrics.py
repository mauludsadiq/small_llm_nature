from llm_nature.metrics import cross_entropy, perplexity

def test_cross_entropy_and_pp():
    pairs = [("a","b"), ("a","b")]
    def prob_fn(x,y): return 0.25
    H = cross_entropy(prob_fn, pairs)
    assert abs(H - (-__import__("math").log(0.25))) < 1e-12
    PP = perplexity(H)
    assert abs(PP - 4.0) < 1e-12
