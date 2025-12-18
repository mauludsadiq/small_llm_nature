from llm_nature.dataset import QAItem
from llm_nature.qa import QAReranker

def test_qa_runs_and_returns_item():
    items = [
        QAItem(q="What is X?", a="X is a thing."),
        QAItem(q="What is Y?", a="Y is a thing."),
    ]
    rr = QAReranker(k=2, alpha=0.5, lam=0.1)
    rr.fit(items)
    ans = rr.answer("What is X?")
    assert ans.a in {"X is a thing.", "Y is a thing."}
